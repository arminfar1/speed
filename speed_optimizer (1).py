"""Optimization class."""

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, Optional, Set, Tuple, Union

import numpy as np
import xpress

from direct_fulfillment_speed.entities.nodes import (
    ODS,
    CarrierType,
    ShipmentType,
    ShippingCarrier,
    Warehouse,
)
from direct_fulfillment_speed.entities.shipment import ShipmentClass
from direct_fulfillment_speed.optimization import solver
from direct_fulfillment_speed.optimization.predict import Predict
from direct_fulfillment_speed.utils import util
from direct_fulfillment_speed.utils.config import ConfigManager

try:
    import amazon_xpress_license
    import xpress as xp
except ImportError:
    import xpress as xp

# Set logger
logger = logging.getLogger()


@dataclass
class ConstraintData:
    lhs: float = 0.0
    total_shipments: float = 0.0
    shipment_types: Set[str] = field(default_factory=set)


class Optimize:
    """Class representing optimization model."""

    def __init__(
        self, shipments_object: ShipmentClass, predict_obj: Predict, config: ConfigManager
    ):
        """Class instances."""
        self.solution: Dict[
            str, Union[float, Dict[Tuple[Union[ODS, Warehouse], str, str], float]]
        ] = {}
        self.shipments_object = shipments_object
        self.predict_obj = predict_obj
        self.config = config
        self.epsilon: float = config.epsilon

        # Model Decision Variables
        self.odsDecisionVars: Dict[
            Union[ODS, Warehouse], Dict[xpress.var, Tuple[float, float]]
        ] = defaultdict(dict)
        self.opt = solver.Solver(f"UPRM", "MINIMIZE")
        self.selectedPad: Dict[Tuple[Union[ODS, Warehouse], str, str], xpress.var] = {}
        self.filtered_pads: Dict[Tuple[Union[ODS, Warehouse], str, str], float] = {}

        # Model configs
        self.min_network_dea = self.config.min_network_dea
        self.min_swa_dea = self.config.min_swa_dea
        self.min_3p_ground_dea = self.config.min_3p_ground_dea
        self.min_3p_air_dea = self.config.min_3p_air_dea
        self.min_unpadded_dea_threshold = self.config.min_unpadded_dea_threshold
        self.gl_list = self.config.get_gl_list
        self._build_dea_targets_cache()

        # Model Inputs
        self.ods_prediction: Dict[
            Union[ODS, Warehouse], Dict[float, float]
        ] = self.predict_obj.get_forecasts

        # Get the total number of shipments across all groups
        self.total_number_swa_shipments: int = (
            self.shipments_object.total_number_shipments_by_group(group=ShippingCarrier.SWA.name)
        )
        self.total_number_3p_ground_shipments: int = (
            self.shipments_object.total_number_shipments_by_group(
                group=CarrierType.THIRD_PARTY.name, shipment_type=ShipmentType.UPS_GROUND
            )
        )
        self.total_number_3p_air_shipments: int = (
            self.shipments_object.total_number_shipments_by_group(
                group=CarrierType.THIRD_PARTY.name, shipment_type=ShipmentType.UPS_AIR
            )
        )

        # Get cumulative shipment percentages
        self.cumulative_shipment_percentages: Dict[
            Union[ODS, Warehouse], float
        ] = self.shipments_object.calculate_cumulative_ship_percentages()

    def build(self):
        """
        Define DVs, Objective and Constraints.
        """
        logger.debug("Creating the decision variables...")
        self.create_decision_variables()
        self.store_decision_variables()
        logger.debug("Creating the objective function...")
        self.create_objective_function()

        logger.debug("Creating the constraints...")
        self.select_only_one_pad_for_ods_const()

        self.create_dea_constraints()

        if self.min_network_dea > 0:
            # Add usdf DEA constraint ONLY if the input DEA is greater than 0.
            self.add_usdf_constraint()

        if self.gl_list:
            # Add GL-level constraint ONLY if the list contains a GL list
            logger.info("A GL list is passed. Creating the GL-level constraints...")
            self.create_gl_level_constraints()

        if self.config.print_lp_file:  # write the lp file
            logger.debug("writing lp file.")
            self.print_lp_file("./")

    def create_decision_variables(self):
        """
        Create the Decision Variables for THIRD_PARTY and SWA groups.
        Returns:
            None.
        """
        # Process THIRD_PARTY group
        self._create_decision_variables(is_third_party=True)

        # Process SWA group
        self._create_decision_variables(is_third_party=False)

    def _create_decision_variables(self, is_third_party):
        """
        Create Decision Variables for the given group.

        Args:
            is_third_party: Flag indicating if the entities are third-party.

        Returns:
            None.
        """
        entity_type = ODS if is_third_party else Warehouse
        min_pad = self.config.min_pad

        for entity, possible_pads in self.ods_prediction.items():
            if isinstance(entity, entity_type):
                if is_third_party and isinstance(entity, ODS):
                    max_pad = (
                        self.config.max_pad
                        if entity.carrier.shipment_type == ShipmentType.UPS_GROUND
                        else self.config.max_pad_air
                    )
                else:
                    max_pad = self.config.max_pad_swa

                self._set_up_decision_variable(
                    entity, possible_pads, min_pad, max_pad, is_third_party
                )

    def _set_up_decision_variable(
        self, entity, possible_pads, min_pad, max_pad, is_third_party
    ) -> None:
        """
        Processes each entity to determine valid decision variables and creates them.
        If the entity's unpadded DEA exceeds a certain threshold, and there is no zero pad in the
        possible pads but there is a positive value, it adjusts the highest negative pad to zero.
        Args:
            entity: The ODS/Warehouse to process.
            possible_pads (dict): Dictionary of quantiles and their corresponding pads.
            min_pad (float): Minimum pad value.
            max_pad (float): Maximum pad value.
            is_third_party (bool): Flag indicating if the entity is third-party.

        Returns:
            None.
        """
        has_valid_decision_variable = False
        unpadded_dea = entity.recent_unpadded_dea
        shipment_type = (
            entity.carrier.shipment_type.name if isinstance(entity, ODS) else ShipmentType.SWA.name
        )

        primary_gl = (
            entity.origin.vendor.vendor_primary_gl
            if isinstance(entity, ODS)
            else entity.vendor.vendor_primary_gl
        )
        min_dea_threshold = self._get_target_dea(
            shipment_type=shipment_type, gl=primary_gl if self.gl_list else ""
        )

        #  # Check if adjustment is needed (i.e., there's no zero pad and there's at least one positive pad)
        if (
            unpadded_dea >= min_dea_threshold
            and 0 not in possible_pads.values()
            and any(pad > 0 for pad in possible_pads.values())
        ):
            self._adjust_negative_pad_to_zero(possible_pads)
            logger.info(f"Adjusted possible pads for {entity}.")

        for quantile, pad in possible_pads.items():
            if self._is_valid_pad(pad, unpadded_dea, min_dea_threshold, min_pad, max_pad):
                has_valid_decision_variable = True
                self._create_decision_var(entity, pad, quantile, is_third_party)

        if not has_valid_decision_variable:
            self._create_default_decision_var(entity, possible_pads, max_pad, is_third_party)

    @staticmethod
    def _adjust_negative_pad_to_zero(possible_pads):
        """
        Adjusts the highest quantile associated with a negative pad to zero. This ensures that
        the optimizer has a neutral (zero) pad option close to the transition point between
        negative and positive pads.

        Args:
            possible_pads (Dict[float, float]): Dictionary of quantiles and their corresponding pad values.
                This represents the predicted pad values across different quantiles.

        Returns:
            None: The method modifies the `possible_pads` dictionary in place, setting the
            highest negative pad value to zero.
        """
        # Sort the pads by quantile
        sorted_pads = sorted(possible_pads.items(), key=lambda x: x[0])

        # Find the highest quantile with a negative pad
        highest_negative_quantile = next(
            (quantile for quantile, pad in reversed(sorted_pads) if pad < 0), None
        )

        # If we found a highest negative quantile, replace it with 0
        if highest_negative_quantile is not None:
            possible_pads[highest_negative_quantile] = 0.0

    @staticmethod
    def _is_valid_pad(pad, unpadded_dea, neg_pad_dea_threshold, min_pad, max_pad):
        """
        Checks if a pad value is valid based on given thresholds and conditions.

        Parameters:
        pad (float): The pad value to check.
        unpadded_dea (float): The unpadded DEA value.
        neg_pad_dea_threshold (float): The negative pad DEA threshold.
        min_pad (float): Minimum pad value.
        max_pad (float): Maximum pad value.

        Returns:
        bool: True if the pad value is valid, False otherwise.
        """
        # Check for None or invalid unpadded_dea
        if pad < 0 and (unpadded_dea == -1.0 or unpadded_dea <= neg_pad_dea_threshold):
            return False

        # Ensure pad is within the allowed range
        if not (min_pad <= pad <= max_pad):
            return False

        return True

    def _create_decision_var(self, entity, pad, quantile, is_third_party):
        """
        Creates a decision variable for an entity.

        Parameters:
        entity: The ODS/Warehouse for which to create the decision variable.
        pad (float): The pad value.
        quantile (float): The quantile value.
        is_third_party (bool): Flag indicating if the entity is third-party carrier.
        """
        decision_var_name, key_tuple, decision_var = self._create_decision_var_and_key_tuple(
            entity, pad, quantile, is_third_party
        )
        self.selectedPad[key_tuple] = decision_var
        self.odsDecisionVars[entity][decision_var] = (quantile, pad)

    def _create_default_decision_var(self, entity, possible_pads, max_pad, is_third_party):
        """
        Creates a default decision variable for an entity when no valid decision variable is found.

        Parameters:
        entity (object): The entity for which to create the default decision variable.
        possible_pads (dict): Dictionary of quantiles and their corresponding pads.
        max_pad (float): Maximum pad value.
        is_third_party (bool): Flag indicating if the entity is third-party.
        """
        logger.warning(
            f"No DV created for {'ODS' if is_third_party else 'warehouse'} {entity}. A default value is used."
        )
        positive_pads = {
            q: min(pad, max_pad) for q, pad in possible_pads.items() if 0 < pad <= max_pad
        }

        if positive_pads:
            closest_quantile = min(positive_pads, key=lambda q: abs(positive_pads[q] - max_pad))
            closest_pad = positive_pads[closest_quantile]
            self._create_decision_var(entity, closest_pad, closest_quantile, is_third_party)
        else:
            self._create_decision_var(entity, 0, 100, is_third_party)

    @staticmethod
    def _create_decision_var_and_key_tuple(entity, pad, quantile, is_third_party):
        """
        Helper method to create decision variable name and key tuple.

        Parameters:
        entity: The ODS/Warehouse to process.
        pad (float): The pad value.
        quantile (float): The quantile value.
        is_third_party (bool): Flag indicating if the entity is third-party carrier.

        Returns:
        tuple: The decision variable name, key tuple, and decision variable.
        """
        pad_str = f"N{abs(pad):.2f}" if pad < 0 else f"{pad:.2f}"
        quant_str = f"{int(quantile)}"
        if is_third_party:
            decision_var_name = f"{entity}_{pad_str.replace('.', 'p')}_{quant_str}"
        else:
            decision_var_name = f"SWA_{entity}_PAD_{pad_str.replace('.', 'p')}_QUAN{quant_str}"
        key_tuple = (
            entity,
            pad_str,
            quant_str,
        )

        decision_var = xp.var(vartype=xp.binary, name=decision_var_name)
        return decision_var_name, key_tuple, decision_var

    def store_decision_variables(self):
        """
        Store decision variables in the solver.
        """
        self.opt.addVariable(self.selectedPad)

    def recent_performance_adjustment(self, pad, entity):
        """
        Calculate the performance adjustment (penalty or incentive) for an entity.

        Parameters:
        pad (float): The pad value.
        entity (object): The entity to adjust.

        Returns:
        float: The performance adjustment value.
        """
        unpadded_dea = entity.recent_unpadded_dea
        shipment_type = (
            entity.carrier.shipment_type.name if isinstance(entity, ODS) else ShipmentType.SWA.name
        )
        primary_gl = (
            entity.origin.vendor.vendor_primary_gl
            if isinstance(entity, ODS)
            else entity.vendor.vendor_primary_gl
        )
        min_dea_threshold = self._get_target_dea(
            shipment_type=shipment_type, gl=primary_gl if self.gl_list else ""
        )

        # Return 0 if min_dea_threshold is None, pad is non-positive, or unpadded_dea is invalid
        if min_dea_threshold == 0.0 or pad <= 0 or unpadded_dea == -1:
            return 0.0

        # Calculate adjustment for low unpadded_dea
        if unpadded_dea <= self.min_unpadded_dea_threshold:
            adjustment = -(1 - unpadded_dea)
            ship_percentage = self.cumulative_shipment_percentages.get(entity, 0)
            return adjustment * ship_percentage / 100

        # Create a lookup for entity type to threshold checks
        entity_checks = {
            Warehouse: unpadded_dea >= min_dea_threshold,
            ODS: unpadded_dea >= min_dea_threshold,  # Simplified ODS check
        }

        # Calculate adjustment based on entity type
        if isinstance(entity, tuple(entity_checks.keys())) and entity_checks[type(entity)]:
            return pad * unpadded_dea

        return 0.0

    def create_objective_function(self) -> None:
        """
        Create the objective function for the optimization problem.
        Returns:
            None.
        """
        total_pads = self._calculate_total_pads()
        total_dea_adjustment = self._calculate_total_dea_adjustment()
        self.opt.addObjective(total_pads + total_dea_adjustment)

    def _calculate_total_pads(self) -> xp.Sum:
        """
        Calculate the total pad cost for the objective function.

        Returns:
            xp.Sum: The total pad cost.
        """
        return xp.Sum(
            self.cumulative_shipment_percentages.get(key, 0) * dv * self._pad_cost(pad)
            for key, dv_quantile_pad_dict in self.odsDecisionVars.items()
            for dv, (quantile, pad) in dv_quantile_pad_dict.items()
        )

    def _pad_cost(self, pad: float) -> float:
        """
        Calculate the cost of a pad value.

        Args:
            pad (float): The pad value.

        Returns:
            float: The pad cost.
        """
        return self.epsilon * pad if pad < 0 else pad

    def _calculate_total_dea_adjustment(self) -> xp.Sum:
        """
        Calculate the total DEA adjustment for the objective function.

        Returns:
            xp.Sum: The total DEA adjustment.
        """
        return xp.Sum(
            dv * self.recent_performance_adjustment(pad, key)
            for key, dv_quantile_pad_dict in self.odsDecisionVars.items()
            for dv, (quantile, pad) in dv_quantile_pad_dict.items()
        )

    def add_usdf_constraint(self):
        """
        Add the USDF DEA constraint to the optimization problem.
        Returns:
            None.
        """
        usdf_dea_constraint_lhs = 0

        for key, dv_quantile_pad_dict in self.odsDecisionVars.items():
            for decision_var, (quantile, pad) in dv_quantile_pad_dict.items():
                probability = quantile / 100.0
                weight = self.cumulative_shipment_percentages.get(key, 0) / 100
                usdf_dea_constraint_lhs += weight * probability * decision_var

        self.opt.addConstraint(
            "DEA_Constraint_USDF",
            "GREATER_EQUAL",
            usdf_dea_constraint_lhs,
            self.min_network_dea,
        )

    def create_gl_level_constraints(self) -> None:
        """
        Create GL-level constraints for the specified GLs in the configuration.
        Returns:
            None.
        """
        if not self.gl_list:
            logger.info("No GL list provided in the configuration. Skipping GL-level constraints.")
            return

        entity_data = self._precompute_entity_data()

        combined_constraints: Dict[Tuple[str, float], ConstraintData] = defaultdict(ConstraintData)

        for entity, (
            gl,
            weight,
            shipment_type,
            target_dea,
            precomputed_terms,
        ) in entity_data.items():
            if target_dea is None:
                logger.error(f"No DEA target found for {shipment_type} and GL {gl}")
                continue

            key = (gl, target_dea)
            constraint_data = combined_constraints[key]
            constraint_data.total_shipments += weight
            constraint_data.shipment_types.add(shipment_type)
            constraint_data.lhs += np.sum(precomputed_terms[:, 0])

        for (gl, target_dea), data in combined_constraints.items():
            if data.total_shipments > 0:
                normalized_lhs = data.lhs / data.total_shipments
                shipment_types_str = "_".join(sorted(data.shipment_types))
                if len(data.shipment_types) > 1:
                    constraint_name = f"GL_DEA_Constraint_{gl}_Merged_{shipment_types_str}"
                else:
                    constraint_name = f"GL_DEA_Constraint_{gl}_{shipment_types_str}"

                self.opt.addConstraint(constraint_name, "GREATER_EQUAL", normalized_lhs, target_dea)
                logger.info(f"Added constraint: {constraint_name} with target DEA {target_dea:.2f}")
            else:
                logger.warning(
                    f"Total shipments for GL {gl} with target DEA {target_dea} is zero. Constraint not added."
                )

    def _get_target_dea(self, shipment_type: str, gl: str = "") -> float:
        """
        Retrieve the target DEA from the cache.

        Args:
            shipment_type (str): The shipment type.
            gl (str): The GL group (optional).

        Returns:
            float: The target DEA value.
        """
        return self.dea_targets_cache.get((shipment_type, gl), 0.0)

    def _build_dea_targets_cache(self):
        """
        Build a cache of DEA targets for all relevant shipment types and GL groups.
        """
        self.dea_targets_cache = {}

        # General DEA targets
        general_targets = {
            ShipmentType.SWA.name: self.min_swa_dea,
            ShipmentType.UPS_GROUND.name: self.min_3p_ground_dea,
            ShipmentType.UPS_AIR.name: self.min_3p_air_dea,
        }

        for shipment_type, min_dea in general_targets.items():
            self.dea_targets_cache[(shipment_type, "")] = min_dea

        # GL-level DEA targets
        if self.gl_list:
            for gl in self.gl_list:
                for shipment_type in [ShipmentType.SWA.name, ShipmentType.UPS_GROUND.name]:
                    # Construct the configuration attribute name
                    config_attr = f"min_dea_{gl.lower()}_{shipment_type.lower()}"
                    min_dea = getattr(self.config, config_attr, 0.0)
                    self.dea_targets_cache[(shipment_type, gl)] = min_dea

    def _precompute_entity_data(self):
        """
        Precompute entity data for GL-level constraints.

        Returns:
            dict: Precomputed entity data.
        """
        entity_data = {}

        for entity, decision_vars in self.odsDecisionVars.items():
            if self._is_entity_valid_gl_group(entity):
                primary_gl = (
                    entity.origin.vendor.vendor_primary_gl
                    if isinstance(entity, ODS)
                    else entity.vendor.vendor_primary_gl
                )
                weight = entity.ship_count
                shipment_type = (
                    entity.carrier.shipment_type.name
                    if isinstance(entity, ODS)
                    else ShipmentType.SWA.name
                )
                target_dea = self._get_target_dea(shipment_type=shipment_type, gl=primary_gl)
                precomputed_terms = np.array(
                    [
                        (weight * (quantile / 100.0) * decision_var, pad)
                        for decision_var, (quantile, pad) in decision_vars.items()
                    ]
                )
                entity_data[entity] = (
                    primary_gl,
                    weight,
                    shipment_type,
                    target_dea,
                    precomputed_terms,
                )

        return entity_data

    def _is_entity_valid_gl_group(self, entity) -> bool:
        """
        Check if the entity is valid for GL-level constraints.
        TODO: For Now, we do not have SWA used for Tires and also no UPS-AIR for furniture's and
        Tires. This may be subject to change.

        Args:
            entity: The ODS/Warehouse.

        Returns:
            bool: True if the entity is valid, False otherwise.
        """
        if isinstance(entity, ODS) and entity.carrier.shipment_type == ShipmentType.UPS_AIR:
            return False
        if isinstance(entity, Warehouse) and entity.primary_gl == "Tires":
            return False
        return entity.primary_gl in self.gl_list

    def add_dea_constraint(self, name_suffix, total_shipments, min_dea):
        """
        Add DEA constraints for the given group.

        Args:
            name_suffix (str): The suffix for the constraint name.
            total_shipments (int): The total number of shipments.
            min_dea (float): The minimum DEA value.
        """
        if total_shipments == 0:
            logger.error("Attempted to add DEA constraint with zero total shipments.")
            return

        dea_constraint_lhs = 0
        constraint_terms = []

        for entity, decision_vars in self.odsDecisionVars.items():
            if name_suffix == ShippingCarrier.SWA.name and isinstance(entity, Warehouse):
                weight = entity.ship_count
                for decision_var, (quantile, pad) in decision_vars.items():
                    probability = quantile / 100.0
                    term = weight * probability * decision_var
                    constraint_terms.append(term)
                    dea_constraint_lhs += term
            elif isinstance(entity, ODS) and entity.carrier.shipment_type.name == name_suffix:
                weight = entity.ship_count
                for decision_var, (quantile, pad) in decision_vars.items():
                    probability = quantile / 100.0
                    term = weight * probability * decision_var
                    constraint_terms.append(term)
                    dea_constraint_lhs += term

        if constraint_terms:
            normalized_lhs = dea_constraint_lhs / total_shipments
            self.opt.addConstraint(
                f"DEA_Constraint_{name_suffix}", "GREATER_EQUAL", normalized_lhs, min_dea
            )
        else:
            logger.warning(f"No positive DEA contribution for {name_suffix}. Constraint not added.")

    def create_dea_constraints(self) -> None:
        """
        Create a DEA for third-part or 1P DEAs.
        Returns:
            None.
        """
        shipment_types = [
            (ShipmentType.UPS_AIR, self.total_number_3p_air_shipments),
            (ShipmentType.UPS_GROUND, self.total_number_3p_ground_shipments),
            (ShipmentType.SWA, self.total_number_swa_shipments),
        ]

        for shipment_type, total_shipments in shipment_types:
            min_dea = self._get_target_dea(shipment_type=shipment_type.name)
            if min_dea > 0.0 and total_shipments > 0:
                self.add_dea_constraint(shipment_type.name, total_shipments, min_dea)

    def select_only_one_pad_for_ods_const(self) -> None:
        """
        Ensure that each ODS or Warehouse can have only one pad value selected.
        Returns:

        """
        for key in self.odsDecisionVars.keys():
            if isinstance(key, ODS):
                constraint_name = f"SelectOnePad_ODS_{key}"
                relevant_decision_vars = [dv for dv, _ in self.odsDecisionVars[key].items()]
            elif isinstance(key, Warehouse):
                constraint_name = f"SelectOnePad_SWA_{key.warehouse_id}"
                relevant_decision_vars = [dv for dv, _ in self.odsDecisionVars[key].items()]
            else:
                logger.warning(f"Unexpected key type: {type(key)}")
                continue

            if relevant_decision_vars:
                self.opt.addConstraint(
                    constraint_name,
                    "EQUAL",
                    xp.Sum(relevant_decision_vars),
                    1,
                )

    def extract_solutions(self):
        """
        Extract the optimization solution and create a dict for them.

        Returns:
            dict: The optimization solution.
        """
        self.solution = {
            "optimization_status": self.opt.getStatus(),
            "objective_value": self.opt.getObjectiveValue(),
            "selected_pads": self.opt.getSolution(self.selectedPad),
        }
        return self.solution

    def filter_selected_pads(self, selected_pads):
        """
        Filter only the selected decision variables i.e., the selected pads.

        Args:
            selected_pads (dict): The selected pads.

        Returns:
            dict: The filtered pads.
        """
        self.filtered_pads = {
            key: value for key, value in selected_pads.items() if value > self.epsilon
        }

    @staticmethod
    def _get_shipment_type(entity):
        """Determine the shipment type for an entity."""
        if isinstance(entity, ODS):
            return entity.carrier.shipment_type.name
        elif isinstance(entity, Warehouse):
            return ShipmentType.SWA.name
        return None

    @property
    def get_dea_constraints_lhs(self) -> Dict[Tuple[str, Optional[str]], float]:
        """
        Get the LHS of the DEA constraints for shipment types and GL-level constraints.
        Only include constraints with target DEA greater than zero.

        Returns:
            Dict[Tuple[str, Optional[str]], float]: The LHS of the DEA constraints.
        """
        dea_constraints_lhs: DefaultDict[Tuple[str, Union[str, None]], float] = defaultdict(float)
        total_shipments: DefaultDict[Tuple[str, Union[str, None]], float] = defaultdict(float)

        for key_tuple, solution_value in self.filtered_pads.items():
            entity, pad_str, quantile = key_tuple
            lhs_contribution = entity.ship_count * (float(quantile) / 100) * solution_value
            shipment_type = self._get_shipment_type(entity)
            primary_gl = entity.primary_gl

            # Keys to consider
            keys_to_check = [(shipment_type, "")]

            # Shipment type constraint key (no GL)

            # GL-level constraint key (if GL group exists)
            if primary_gl:
                keys_to_check.append((shipment_type, primary_gl))

            for a_key in keys_to_check:
                min_dea = self.dea_targets_cache.get(a_key, 0.0)
                if min_dea > 0.0:
                    dea_constraints_lhs[a_key] += lhs_contribution
                    total_shipments[a_key] += entity.ship_count

        for key in list(dea_constraints_lhs.keys()):
            if total_shipments[key] > 0:
                dea_constraints_lhs[key] /= total_shipments[key]
            else:
                logger.warning(f"Total shipments for {key} is zero. Cannot normalize LHS.")

        logger.debug(f"Computed DEA LHS values are {dict(dea_constraints_lhs)}")
        return dict(dea_constraints_lhs)

    @property
    def get_average_speed(self) -> Dict[Tuple[str, Optional[str]], float]:
        """
        Calculate the average speed as the average of (c2p_unpadded + pad) for
        shipment types and GL-level constraints, excluding constraints with zero target DEA.

        Returns:
           Dict[Tuple[str, Optional[str]], float]: The average speed for each constraint key.
        """
        total_speed: DefaultDict[Tuple[str, Optional[str]], float] = defaultdict(float)
        total_shipments: DefaultDict[Tuple[str, Optional[str]], int] = defaultdict(int)

        for key_tuple, solution_value in self.filtered_pads.items():
            entity, pad_str, quantile = key_tuple
            pad = float(pad_str.replace("N", "-").replace("p", "."))
            shipment_type = self._get_shipment_type(entity)
            primary_gl = entity.primary_gl

            # Skip entities with no shipment type
            if shipment_type is None:
                continue

            # Keys to consider
            keys_to_check = [(shipment_type, "")]

            # GL-level constraint key (if GL group exists)
            if primary_gl:
                keys_to_check.append((shipment_type, primary_gl))

            # Retrieve shipments for the entity
            shipments = self.shipments_object.get_shipments_for_entity(entity)

            for shipment in shipments:
                speed = shipment.c2p_days_unpadded + pad

                for a_key in keys_to_check:
                    min_dea = self.dea_targets_cache.get(a_key, 0.0)
                    if min_dea > 0.0:
                        total_speed[a_key] += speed
                        total_shipments[a_key] += 1

        # Calculate average speeds
        average_speed: Dict[Tuple[str, Optional[str]], float] = {}
        for key in list(total_speed.keys()):
            if total_shipments[key] > 0:
                average_speed[key] = total_speed[key] / total_shipments[key]
            else:
                logger.warning(
                    f"Total shipments for {key} is zero. Cannot calculate average speed."
                )

        logger.debug(f"Average speed: {average_speed}")
        return average_speed

    def print_lp_file(self, output_path):
        """This will print the lp file of the model."""
        self.opt.write(output_path + util.date_now())

    def solve(self):
        """
        Solve the optimization problem.
        Returns:
            None.

        """
        try:
            self.set_controls()

            logger.info(f"Building the model ...")
            self.build()
            logger.info(f"Building the model is done.")

            logger.info(f"Optimizing ...")
            self.opt.mip_optimize()

            solution = self.extract_solutions()
            self.filter_selected_pads(solution["selected_pads"])
            logger.info(f"Extracting the solution is done.")

        except IOError:
            logger.error(f"\tERROR: No solution for the problem!")
            sys.exit()

    def set_controls(self):
        """
        Set all the algorithm controls in XPRESS.
        Returns:
            None.
        """
        self.opt.setControl("maxtime", -self.config.xpress_max_solve)
        self.opt.setControl("outputlog", self.config.xpress_outputflag)
        self.opt.setControl("miprelstop", self.config.integrality_gap_percentage)
        self.opt.setControl("mippresolve", self.config.xpress_presolve)

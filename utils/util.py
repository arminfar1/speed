from datetime import datetime, timedelta, timezone


def get_clockwise_time_diff(start_time: datetime, end_time: datetime):
    """Get the time difference between time units without date, by counting clockwise."""
    if start_time > end_time:
        # In the provided fight schedule, there is a chance the end_time is earlier than start_time
        # since dates are not provided.
        end_time += timedelta(days=1)
    return (end_time - start_time).total_seconds()


def parse_s3_path(s3_path):
    """
    Parses the S3 path to extract the bucket name and object key.

    :param s3_path: The full S3 path (e.g., s3://bucket-name/object-key)
    :return: A tuple containing the bucket name and object key
    """
    if not s3_path.startswith("s3://"):
        raise ValueError("Invalid S3 path format. Must start with 's3://'")

    # Remove the 's3://' prefix and split the remaining string by the first '/'
    path_without_scheme = s3_path[5:]
    bucket_name, object_key = path_without_scheme.split("/", 1)

    return bucket_name, object_key


def concat_strings(separator: str, *argv):
    """Create a concatenation of arguments."""
    return str(separator).join(str(arg) for arg in argv)


def date_now(include_time=False):
    """
    Return date and time now, with optional formatting.
    """
    utc_datetime = datetime.now(tz=timezone.utc)

    if include_time:
        return utc_datetime.strftime("%Y-%m-%d %H%M%S")
    else:
        return utc_datetime.strftime("%Y-%m-%d")

import hashlib
import json


def hash_list_of_strings(lst):
    # Sort the list to normalize order
    sorted_strings = sorted(lst)
    # Join them into a single string and hash it
    combined = ''.join(sorted_strings)
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()
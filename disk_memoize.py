import os
import pickle
import hashlib
from functools import wraps

def disk_memoize(cache_file):
    def decorator(func):
        # Load cache if it exists
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                print(f"CACHE HIT {cache_file}")
                cache = pickle.load(f)
        else:
            cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key from function arguments
            key_data = (func.__name__, args, tuple(sorted(kwargs.items())))
            key = hashlib.sha256(pickle.dumps(key_data)).hexdigest()

            if key in cache:
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result

            # Save cache to disk
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)

            return result

        return wrapper
    return decorator
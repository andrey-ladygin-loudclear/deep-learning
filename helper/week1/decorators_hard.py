def logger(filename):
    def decorator(func):
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            with open(filename, 'w') as f:
                f.write(str(result))
            return result
        return wrapped
    return decorator

@logger('new_log.txt')
def summator(num_list):
    return sum(num_list)

summator([1,2,3,4,5])
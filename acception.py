import random

def naive_accept(seed:int=42, acceptance_ratio:float=0.0) -> bool:
    '''
    This function randomly decides whether a courier will accept a parcel according to the acceptance ratio.
    It denies all the ratio values that < 0.1.

    Parameters:
    seed: int
        Random seed.
    acceptance_ratio: float
        A courier's acceptance probability for a parcel.

    Return:
    A boolean value indicating whether the courier will accept the parcel.
    '''
    random.seed(seed)

    # Ignore acceptance ratios that are too small.
    if acceptance_ratio < 0.1:
        return False
    
    random_num = random.randint(0, 1000) # If the random value is in the range calculated by acceptance, then the courier will accept the parcel.
    if random_num <= acceptance_ratio * 1000:
        return True
    else:
        return False
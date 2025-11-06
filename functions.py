def calculate_gpu_cost(hours, rate_per_hour):
    total = hours * rate_per_hour
    return round(total, 2)


def check_budget(cost, budget=100):
    if cost > budget:
        return f"Over budget by ${cost - budget}"
    else:
        return f"Within budget. ${budget - cost} remaining"


# Test the functions
gpu_hours = 10
hourly_rate = 0.69
total_cost = calculate_gpu_cost(gpu_hours, hourly_rate)
print(f"GPU Cost: ${total_cost}")
print(check_budget(total_cost))

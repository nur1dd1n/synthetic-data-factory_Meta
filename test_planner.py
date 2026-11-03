from app.services.generation_planner import create_generation_plan

profile = {
    "rows": 300,
    "columns_count": 3,
    "duplicate_rows": 0,
    "columns": [
        {
            "name": "amount",
            "dtype": "float64",
            "missing_count": 0,
            "missing_percent": 0,
            "unique_count": 250,
            "statistics": {
                "mean": 120.5,
                "std": 40.2,
                "min": 1.0,
                "max": 500.0,
            },
        },
        {
            "name": "category",
            "dtype": "object",
            "missing_count": 0,
            "missing_percent": 0,
            "unique_count": 4,
        },
        {
            "name": "fraud",
            "dtype": "int64",
            "missing_count": 0,
            "missing_percent": 0,
            "unique_count": 2,
        },
    ],
}

plan = create_generation_plan(
    user_request="Generate synthetic transactions while preserving fraud patterns.",
    profile=profile,
    target_column="fraud",
    task="classification",
)

print("\nMETA AI GENERATION PLAN\n")
print(plan)






# print("1. Test started")

# from app.services.generation_planner import create_generation_plan

# print("2. Planner imported")

# profile = {
#     "rows": 300,
#     "columns_count": 2,
#     "duplicate_rows": 0,
#     "columns": [
#         {
#             "name": "amount",
#             "dtype": "float64",
#             "missing_count": 0,
#             "missing_percent": 0,
#             "unique_count": 250,
#         },
#         {
#             "name": "fraud",
#             "dtype": "int64",
#             "missing_count": 0,
#             "missing_percent": 0,
#             "unique_count": 2,
#         },
#     ],
# }

# print("3. Creating plan...")

# plan = create_generation_plan(
#     user_request="Generate synthetic transactions.",
#     profile=profile,
#     target_column="fraud",
#     task="classification",
# )

# print("4. Meta AI response:")
# print(plan)
# print("TEST FILE IS RUNNING", flush=True)

# import time

# time.sleep(2)

# print("TEST FINISHED", flush=True)
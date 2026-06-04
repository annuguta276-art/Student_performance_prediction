import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "study_hours": np.random.randint(1, 11, n),
    "sleep_hours": np.random.randint(4, 10, n),
    "stress_level": np.random.randint(1, 6, n),
    "attendance": np.random.randint(50, 101, n),
    "focus_level": np.random.randint(1, 6, n),
    "motivation_level": np.random.randint(1, 6, n),
    "social_media_usage": np.random.randint(0, 8, n),
    "revision_frequency": np.random.randint(1, 6, n),
    "concentration_level": np.random.randint(1, 6, n),
    "class_attention": np.random.randint(1, 6, n)
})

data["marks"] = (
    data["study_hours"] * 5 +
    data["attendance"] * 0.4 +
    data["focus_level"] * 4 +
    data["motivation_level"] * 3 +
    data["revision_frequency"] * 3 +
    data["concentration_level"] * 3 +
    data["class_attention"] * 3 -
    data["stress_level"] * 2 -
    data["social_media_usage"] * 2
)

data["marks"] += np.random.normal(0, 5, n)

data["marks"] = data["marks"].clip(0, 100)

data.to_csv("student_data.csv", index=False)

print("Dataset Created Successfully")
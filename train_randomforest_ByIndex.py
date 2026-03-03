import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/all_features_dataset.csv")
# df = pd.read_csv("data/all_raw_features_dataset.csv")

# Make sure there's a column for video identification
print(df.columns)

# Define train/test videos
########### 50% train 50% test ##############
# train_videos = [
#     'lifting_front_normal_1_landmarks', 'lifting_side_normal_2_landmarks',
#     'squatting_front_normal_1_landmarks', 'squatting_side_normal_3_landmarks',
#     'overhead_front_normal_1_landmarks', 'overhead_side_normal_2_landmarks'
# ]
# test_videos = [
#     'lifting_front_normal_3_landmarks', 'lifting_side_normal_4_landmarks',
#     'squatting_front_normal_2_landmarks', 'squatting_side_normal_4_landmarks',
#     'overhead_front_normal_3_landmarks', 'overhead_side_normal_4_landmarks'
# ]
############ 75% train 25% test (front view test) ##############
# train_videos = [
#     'lifting_front_normal_1_landmarks', 'lifting_side_normal_2_landmarks',
#     'lifting_side_normal_4_landmarks', 'squatting_side_normal_4_landmarks',
#     'overhead_side_normal_4_landmarks',
#     'squatting_front_normal_1_landmarks', 'squatting_side_normal_3_landmarks',
#     'overhead_front_normal_1_landmarks', 'overhead_side_normal_2_landmarks'
# ]
# test_videos = [
#     'lifting_front_normal_3_landmarks', 
#     'squatting_front_normal_2_landmarks', 
#     'overhead_front_normal_3_landmarks'
# ]
############ 75% train 25% test (side view test) ##############
train_videos = [
    'lifting_front_normal_1_landmarks', 'lifting_side_normal_2_landmarks',
    'lifting_front_normal_3_landmarks','squatting_front_normal_2_landmarks',
    'overhead_front_normal_3_landmarks', 
    'squatting_front_normal_1_landmarks', 'squatting_side_normal_3_landmarks',
    'overhead_front_normal_1_landmarks', 'overhead_side_normal_2_landmarks'
]
test_videos = [
     'lifting_side_normal_4_landmarks',
     'squatting_side_normal_4_landmarks',
    'overhead_side_normal_4_landmarks'
]

# train_videos = [
#     'lifting_front_normal_1_landmarks', 'lifting_side_normal_2_landmarks',
#     'lifting_side_normal_4_landmarks', 'squatting_side_normal_4_landmarks',
#     'overhead_side_normal_4_landmarks',  'squatting_front_normal_2_landmarks', 
#     'lifting_front_normal_3_landmarks','overhead_front_normal_3_landmarks',
#      'squatting_side_normal_3_landmarks',
#     'overhead_front_normal_1_landmarks', 'overhead_side_normal_2_landmarks'
# ]
# test_videos = [
     
# 'squatting_front_normal_1_landmarks',
# ]

# Split the data by video name
train_df = df[df['video_name'].isin(train_videos)]
test_df = df[df['video_name'].isin(test_videos)]

# Separate features and labels
X_train = train_df.drop(columns=['movement', 'video_name'])
y_train = train_df['movement']

X_test = test_df.drop(columns=['movement', 'video_name'])
y_test = test_df['movement']

# Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

import joblib
joblib.dump(clf, "models/randomforest_byIndex_SideView.pkl")
print("Model saved to models/randomforest_byIndex_SideView.pkl")
# Evaluation
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=['lifting', 'overhead', 'squatting'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', xticklabels=['lifting', 'overhead', 'squatting'], yticklabels=['lifting', 'overhead', 'squatting'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Split by Video")
plt.show()

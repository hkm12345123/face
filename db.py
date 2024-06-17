import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://attendance-system-40237-default-rtdb.asia-southeast1.firebasedatabase.app/"
    },
)

studentRef = db.reference("Students")

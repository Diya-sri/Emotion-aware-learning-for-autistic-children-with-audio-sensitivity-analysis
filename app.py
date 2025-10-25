from flask import Flask, render_template, request, redirect, url_for, session,Response
import cv2
import numpy as np
import tensorflow as tf
import time  
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from keras.models import load_model
import os
import ssl
from flask import Flask, Response, jsonify
from collections import Counter

app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model1 = tf.keras.models.load_model('autism_model.h5')
class_label = ['anger', 'fear', 'joy', 'natural', 'sadness', 'surprise']

email_sent = False
emotion_history = []
emotion_capture_threshold = 10  
frame_counter = 0
time_window = 100
dominant_emotion = ""
latest_emotion = dominant_emotion

model = tf.keras.models.load_model('model1.h5')

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        total_score = sum(int(request.form.get(f'q{i}', 0)) for i in range(1, 10))
        result = detect_depression(total_score)
        return render_template('result.html', score=total_score, result=result)
    return render_template('index.html')

@app.route('/camera')
def camera():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    class_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']

    def preprocess_frame(face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  
        face = cv2.resize(face, (224, 224))  
        face = np.expand_dims(face, axis=-1)  
        face = np.repeat(face, 3, axis=-1)  
        face = face / 255.0  
        face = np.expand_dims(face, axis=0)  
        return face

    def predict_emotion(face):
        img_array = preprocess_frame(face)
        prediction = model.predict(img_array)  
        predicted_class_index = np.argmax(prediction, axis=1)[0]  
        predicted_class_label = class_labels[predicted_class_index]
        confidence = np.max(prediction)  
        return predicted_class_label, confidence

    detected_emotions = []
    start_time = time.time()  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_3channel = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        faces = face_cascade.detectMultiScale(gray_frame_3channel, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            emotion, confidence = predict_emotion(face)

            detected_emotions.append((emotion, confidence))

            label = f"{emotion} ({confidence*100:.2f}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imshow('Emotion Detection', frame)

        if time.time() - start_time > 120:  
            break  

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if detected_emotions:
        emotions, confidences = zip(*detected_emotions)
        most_common_emotion = max(set(emotions), key=emotions.count)
        avg_confidence = sum(confidences) / len(confidences)
    else:
        most_common_emotion = "No face detected"
        avg_confidence = 0.0

    session['emotion'] = most_common_emotion
    session['confidence'] = round(avg_confidence * 100, 2)

    return redirect(url_for('emotion_result'))

emotion_emoji_map = {
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happiness": "😃",
    "neutrality": "😐",
    "sadness": "😢",
    "surprise": "😲"
}

@app.route('/emotion_result')
def emotion_result():
    emotion = session.get('emotion', "Unknown")
    emoji = emotion_emoji_map.get(emotion, "❓")  
    return render_template('emotion_result.html', emotion=emotion.capitalize(), emoji=emoji)





def send_feedback_email(subject, feedback):
    sender_email = "triossoftwaremail@gmail.com"
    receiver_email = "mathukarthik2225@gmail.com"
    password = "knaxddlwfpkplsik"  

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    msg.attach(MIMEText(feedback, "plain"))  

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Feedback email sent successfully!")
        return True
    except Exception as e:
        print("Error sending feedback email:", e)
        return False


        

@app.route('/autism')
def autism():
    session['attempts'] = 0
    session['correct'] = 0
    return render_template('autism.html')


@app.route("/submit", methods=["POST"])
def submit_feedback():
    data = request.get_json()
    clicked_emotion = data.get("emotion")
    detected_emotion = session.get('emotion', "Unknown")

    if "attempts" not in session:
        session['attempts'] = 0
        session['correct'] = 0

    session['attempts'] += 1

    if clicked_emotion == detected_emotion:
        session['correct'] += 1

    if session['attempts'] == 3:
        subject = ""
        feedback = ""
        if session['correct'] == 3:
            subject = "Learning Well"
            feedback = "User correctly identified the emotion 3 times."
        elif session['correct'] == 2:
            subject = "Learning Much Better"
            feedback = "User correctly identified the emotion 2 times."
        elif session['correct'] == 0:
            subject = "Learning Not Well"
            feedback = "User incorrectly identified the emotion 3 times."

        send_feedback_email(subject, feedback)  

        session['attempts'] = 0  
        session['correct'] = 0

    return jsonify({"message": "Feedback received."})



def preprocess_frame(frame):
    """Preprocess the face before feeding it to the model."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        return face, (x, y, w, h)

    return None, None

def predict_emotion(frame):
    """Predict the emotion from the detected face."""
    processed_face, face_coords = preprocess_frame(frame)
    if processed_face is not None and model1 is not None:
        prediction = model1.predict(processed_face)[0]
        emotion_index = np.argmax(prediction)
        emotion = class_label[emotion_index]
        confidence = float(prediction[emotion_index])
        return emotion, confidence, face_coords
    return "", 0.0, None  

def send_email(image_path, detected_emotion):
    """Send an email with the captured emotion image and detected emotion text."""
    sender_email = "triossoftwaremail@gmail.com"
    receiver_email = "mathukarthik2225@gmail.com"
    password = "knaxddlwfpkplsik"

    subject = f"Emotion Detected: {detected_emotion}"
    body = f"The detected dominant emotion is: {detected_emotion}\n\nAttached is the captured frame."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, "plain"))

    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename=emotion.jpg")
        msg.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

    global email_sent
    email_sent = True  

tracked_faces = {}  

def generate_frames():
    """Capture video frames, analyze emotions, and send email for the most dominant emotion."""
    global email_sent, frame_counter
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            emotion, confidence, face_coords = predict_emotion(frame)

            if emotion:
                emotion_history.append(emotion)

                label = f"{emotion} ({confidence*100:.2f}%)"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame_counter += 1

        if frame_counter >= time_window and not email_sent:
            if emotion_history:
                dominant_emotion = Counter(emotion_history).most_common(1)[0][0]  

                image_path = "dominant_emotion.jpg"
                cv2.imwrite(image_path, frame)
                send_email(image_path, dominant_emotion)
                email_sent = True  

            emotion_history.clear()
            frame_counter = 0
            email_sent = False  

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

        
@app.route('/video_feed')
def video_feed():
    """Stream the live video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion_data')
def emotion_data():
    """Return latest detected emotion data as JSON."""
    return jsonify(latest_emotion)



@app.route('/audio', methods=['GET'])
def audio():
    return render_template('audio.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle audio file upload and save it."""
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['audio']
    audio_path = "static/uploads/" + audio_file.filename
    audio_file.save(audio_path)

    return jsonify({"audio_url": f"/{audio_path}"})





if __name__ == '__main__':
    app.run(debug=False, port=700)

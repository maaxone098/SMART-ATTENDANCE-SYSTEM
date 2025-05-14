import os
# --- Force TensorFlow CPU ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import config
import cv2
import numpy as np
import pickle
import time
import pandas as pd
import smtplib
import ssl
from email.message import EmailMessage
import threading
from ultralytics import YOLO
from deepface import DeepFace
# Import verification module to access distance functions (Fifth attempt)
try:
    from deepface.modules import verification
    print("Imported verification module from deepface.modules")
    # Define functions locally for easier use in find_distance helper
    findCosineDistance = verification.find_cosine_distance
    findEuclideanDistance = verification.find_euclidean_distance
    l2_normalize = verification.l2_normalize
except ImportError as e_verify:
    print("FATAL ERROR: Could not import DeepFace verification module or its functions.")
    print(f"Import Error: {e_verify}")
    print("Please check your DeepFace installation and version.")
    exit()
except AttributeError as ae:
    print(f"FATAL ERROR: Could not find expected distance functions in deepface.modules.verification.")
    print(f"Attribute Error: {ae}")
    print("Please check your DeepFace installation and version.")
    exit()


# --- Configuration ---
YOLO_MODEL_PATH = "best.pt"
ENCODING_FILE = "encodings_deepface.pkl"
STUDENT_CSV = "students_info/student_data.csv"

# --- OpenCV DNN Face Detector Configuration ---
PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD_DNN = 0.5

# --- DeepFace Configuration ---
DEEPFACE_MODEL_NAME = "VGG-Face"
DISTANCE_METRIC = "cosine" # Options: 'cosine', 'euclidean', 'euclidean_l2'
RECOGNITION_THRESHOLD = 0.50 # Adjust based on testing (Cosine: lower is better)

# --- Camera Configuration ---
CAMERA_INDEX = 1 # 0 for default, 1, 2 for others, or RTSP/HTTP URL string
DESIRED_WIDTH = 1280 # Request camera width (e.g., 1280 for 720p)
DESIRED_HEIGHT = 720  # Request camera height (e.g., 720 for 720p)

# --- Display Window Configuration ---
DISPLAY_WIDTH = 960 # Width of the displayed window (pixels)

# --- Attendance & Feature Flags ---
ATTENDANCE_SESSION_DURATION = 2 * 60 
REQUIRED_RECOGNITION_DURATION = 1 * 60 
HR_SESSION_ACTIVE = True
SNAPSHOT_INTERVAL = 1 * 60 
RECORDING_FPS = 30 # Set desired recording FPS

# --- Output Directories ---
SNAPSHOT_DIR = "attendance_info/snapshots"
RECORDINGS_DIR = "attendance_info/recordings"
REPORTS_DIR = "attendance_info/reports"

# --- Email Configuration ---
EMAIL_SENDER = config.EMAIL_SENDER
EMAIL_PASSWORD = config.EMAIL_PASSWORD
EMAIL_SERVER = "smtp.gmail.com"
EMAIL_PORT = 465

# --- Create Output Directories ---
print("Creating output directories if they don't exist...")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Load YOLO Model ---
print("Loading YOLOv8 model...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model from {YOLO_MODEL_PATH}: {e}")
    exit()

# --- Load Face Encodings (Embeddings) ---
print(f"Loading known face embeddings from {ENCODING_FILE}...")
try:
    with open(ENCODING_FILE, "rb") as f:
        data = pickle.load(f)
        known_embeddings_list = data["embeddings"]
        known_names = data["names"]
        if not known_names:
            print("Error: No names found in encoding file.")
            exit()
        known_embeddings_np = np.array(known_embeddings_list)
        # Ensure names used as keys are consistent (e.g., stripped)
        unique_stripped_names = sorted(list(set(name.strip() for name in known_names)))
        print(f"Loaded {len(unique_stripped_names)} unique known names.")
        print(f"Total embeddings loaded: {len(known_embeddings_list)}") 
except FileNotFoundError:
    print(f"Error: Encoding file not found at {ENCODING_FILE}")
    exit()
except Exception as e:
    print(f"Error loading encoding file: {e}")
    exit()

# --- Load Student Data (for email lookup) ---
print(f"Loading student data from {STUDENT_CSV}...")
student_info_map = {} # Store a dictionary for each student: {'StudentEmailAddress': ..., 'ParentEmailAddress': ...}
df_students = None
try:
    df_students = pd.read_csv(STUDENT_CSV, skipinitialspace=True)
    df_students.columns = df_students.columns.str.strip() # Strip header spaces
    # Ensure required columns exist
    required_cols = ['Name', 'StudentEmailAddress', 'ParentEmailAddress']
    if not all(col in df_students.columns for col in required_cols):
        print(f"Error: {STUDENT_CSV} must contain columns: {', '.join(required_cols)}")
        df_students = None # Invalidate df_students to prevent further errors
    else:
        df_students['Name'] = df_students['Name'].str.strip()
        for index, row in df_students.iterrows():
            name = row['Name']
            student_info_map[name] = {
                'StudentID': row.get('StudentID', 'N/A'), # Handle if StudentID is missing
                'StudentEmailAddress': row['StudentEmailAddress'],
                'ParentEmailAddress': row['ParentEmailAddress']
            }
        print(f"Loaded data for {len(student_info_map)} students from CSV.")

except FileNotFoundError:
    print(f"Warning: {STUDENT_CSV} not found. Cannot look up StudentID/Email.")
    df_students = None
except Exception as e:
    print(f"Error loading student data from {STUDENT_CSV}: {e}")
    df_students = None


# --- Load OpenCV DNN Model (Optional Cropping) ---
print("Loading OpenCV DNN face detector model (for cropping)...")
try:
    face_detector_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("OpenCV DNN model loaded successfully.")
except cv2.error as e:
    print(f"Warning: Could not load OpenCV DNN model: {e}. Will rely on DeepFace internal detector.")
    face_detector_net = None

# --- Initialize Attendance ---
attendance_status = {name: 'Absent' for name in unique_stripped_names}
# Initialize duration tracking
recognition_duration = {name: 0.0 for name in unique_stripped_names} 
print(f"Initialized attendance for {len(attendance_status)} unique students.")

# --- Initialize Snapshot Variables ---
if HR_SESSION_ACTIVE:
    last_snapshot_time = time.time()

# --- Initialize Video Capture ---
print(f"Attempting to open video capture device: {CAMERA_INDEX}")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open video capture device {CAMERA_INDEX}.")
    exit()

# --- Set Desired Resolution ---
print(f"Requesting camera resolution: {DESIRED_WIDTH}x{DESIRED_HEIGHT}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

# --- Get Actual Resolution and FPS ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_fps = cap.get(cv2.CAP_PROP_FPS)
actual_recording_fps = RECORDING_FPS
if cam_fps > 0 and cam_fps < RECORDING_FPS:
    print(f"Warning: Camera reported FPS ({cam_fps}) is lower than desired recording FPS ({RECORDING_FPS}). Using camera FPS for recording.")
    actual_recording_fps = cam_fps
elif cam_fps == 0:
     print(f"Warning: Camera did not report FPS. Using configured recording FPS ({RECORDING_FPS}).")

print(f"Actual camera resolution set to: {frame_width}x{frame_height}")
print(f"Using recording FPS: {actual_recording_fps}")


# --- Initialize Video Writer ---
timestamp_vid = time.strftime("%Y-%m-%d_%I%M%S%p")
video_filename = os.path.join(RECORDINGS_DIR, f"attendance_recording_{timestamp_vid}.mp4") 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video_writer = cv2.VideoWriter(video_filename, fourcc, actual_recording_fps, (frame_width, frame_height))
if not video_writer.isOpened():
    print(f"Error: Could not open video writer for {video_filename} using mp4v codec.")
    print("Ensure necessary video codecs (e.g., H.264 via ffmpeg or gstreamer) are installed and accessible by OpenCV.")
    print("Consider switching back to 'XVID' codec and '.avi' extension if this persists.")
    video_writer = None
else:
    print(f"Recording video to {video_filename} at {actual_recording_fps} FPS target.")


session_start_time = time.time()
previous_frame_time = session_start_time 

# --- Helper Function for Distance Calculation ---
def find_distance(emb1, emb2, metric=DISTANCE_METRIC):
    """Calculates distance between two embeddings using the specified metric."""
    try:
        if metric == 'cosine':
            return findCosineDistance(emb1, emb2)
        elif metric == 'euclidean':
            return findEuclideanDistance(emb1, emb2)
        elif metric == 'euclidean_l2':
            return findEuclideanDistance(l2_normalize(emb1), l2_normalize(emb2))
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
    except NameError:
        print("\nFATAL ERROR: DeepFace distance functions were not defined correctly.")
        return float('inf')
    except Exception as e:
        print(f"\nError during distance calculation: {e}")
        return float('inf')


# --- Helper Function for Snapshots ---
def take_snapshot(frame_to_save):
    """Saves the current frame as a snapshot."""
    timestamp = time.strftime("%Y-%m-%d_%I%M%S%p") 
    filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
    try:
        cv2.imwrite(filename, frame_to_save)
        print(f"\nSnapshot saved: {filename}")
    except Exception as e:
        print(f"\nError saving snapshot: {e}")


# --- Main Loop ---
print("Starting real-time attendance...")
while True:
    # --- Calculate time elapsed since last frame processing ---
    current_frame_time = time.time()
    time_elapsed = current_frame_time - previous_frame_time

    ret, frame = cap.read()
    if not ret:
        if isinstance(CAMERA_INDEX, str):
             print("End of video file reached.")
             break
        else:
            print("Error: Failed to grab frame from camera.")
            break

    # --- Face Detection with YOLOv8 ---
    yolo_results = yolo_model(frame, verbose=False)

    recognized_names_this_frame = set() 
    if len(yolo_results) > 0 and hasattr(yolo_results[0], 'boxes'):
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = yolo_results[0].boxes.conf.cpu().numpy()
    else:
        boxes = []
        confs = []

    # --- Process Each Detected Face ---
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        conf = confs[i]

        pad = 5
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(frame_width, x2 + pad)
        crop_y2 = min(frame_height, y2 + pad)

        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        name = "Unknown"
        color = (0, 0, 255)

        if face_crop.size == 0:
            continue

        try:
            # --- Generate Embedding for the Detected Face using DeepFace ---
            embedding_objs = DeepFace.represent(
                img_path = face_crop,
                model_name = DEEPFACE_MODEL_NAME,
                enforce_detection = True,
                detector_backend = 'opencv'
            )

            if embedding_objs and isinstance(embedding_objs, list) and len(embedding_objs) > 0 and 'embedding' in embedding_objs[0]:
                current_embedding = np.array(embedding_objs[0]['embedding'])

                # --- Compare with Known Embeddings ---
                min_dist = float('inf')
                best_match_index = -1

                for j, known_embedding in enumerate(known_embeddings_np):
                    dist = find_distance(current_embedding, known_embedding, metric=DISTANCE_METRIC)
                    if dist < min_dist:
                        min_dist = dist
                        best_match_index = j

                # --- Check if Match Exceeds Threshold ---
                if best_match_index != -1 and min_dist < RECOGNITION_THRESHOLD:
                    # Get name from known_names list, ensuring it's stripped of spaces
                    recognized_name = known_names[best_match_index].strip()
                    name = recognized_name 
                    color = (0, 255, 0)
                    recognized_names_this_frame.add(recognized_name) 

                    # --- Update Duration and Check Threshold ---
                    if recognized_name in recognition_duration:
                        recognition_duration[recognized_name] += time_elapsed
                        # Check if threshold is met AND student is currently Absent
                        if (attendance_status[recognized_name] == 'Absent' and
                            recognition_duration[recognized_name] >= REQUIRED_RECOGNITION_DURATION):
                            attendance_status[recognized_name] = 'Present'
                            print(f"\n>> {recognized_name} marked PRESENT.")
                    else:
                        # This might happen if known_names had duplicates/inconsistencies
                        # Ensure initialization uses unique_stripped_names
                        print(f"Warning: Recognized name '{recognized_name}' not in duration tracking list.")

        except ValueError as ve:
            # print(f"Debug: DeepFace ValueError on crop: {ve}")
            pass
        except Exception as e:
            print(f"\nError during DeepFace processing or comparison: {e}")

        # --- Draw Bounding Box and Name ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10
        cv2.rectangle(frame, (x1, text_y - text_height - baseline), (x1 + text_width, text_y + baseline), color, cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)


    # --- Display Recognized Count (based on names recognized *in this frame*) ---
    count_text = f"Recognized Now: {len(recognized_names_this_frame)}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # --- Display Frame ---
    aspect_ratio = frame_height / frame_width if frame_width > 0 else 1
    display_height = int(DISPLAY_WIDTH * aspect_ratio)
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, display_height), interpolation=cv2.INTER_AREA)
    cv2.imshow('Attendance System', display_frame)

    # --- Write the *original* frame to Video ---
    if video_writer is not None:
        try:
            video_writer.write(frame)
        except Exception as e:
            print(f"\nError writing frame to video: {e}")

    # --- Snapshot Logic ---
    current_time_snapshot = time.time() 
    if HR_SESSION_ACTIVE and (current_time_snapshot - last_snapshot_time >= SNAPSHOT_INTERVAL):
         snapshot_thread = threading.Thread(target=take_snapshot, args=(frame.copy(),))
         snapshot_thread.daemon = True
         snapshot_thread.start()
         last_snapshot_time = current_time_snapshot 

    # --- Update previous frame time for next iteration's time_elapsed calculation ---
    previous_frame_time = current_frame_time

    # --- Check for Session Timeout ---
    if time.time() - session_start_time > ATTENDANCE_SESSION_DURATION:
        print(f"\nAttendance session duration ({ATTENDANCE_SESSION_DURATION / 60.0:.1f} minutes) reached. Exiting...")
        break 

    # --- Exit Condition (Manual Quit) ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting application manually...")
        break

# --- Cleanup ---
cap.release()
if video_writer is not None:
    video_writer.release()
    print(f"Video recording saved to {video_filename}")
cv2.destroyAllWindows()
print("Video capture released and windows closed.")

# --- Generate Excel Report ---
print("\nGenerating attendance report...")
attendance_list = []
# Use the initial list of unique stripped names to ensure all are reported
print(f"Generating report for {len(unique_stripped_names)} unique names found in encodings file.")

for name in unique_stripped_names:
     status = attendance_status.get(name, 'Absent')
     student_id = 'N/A'
     student_email = 'N/A' # Renamed for clarity
     parent_email = 'N/A'  # New variable
     found_in_csv = False

     if name in student_info_map: # Use the new map
         info = student_info_map[name]
         student_id = info.get('StudentID', 'N/A')
         student_email = info.get('StudentEmailAddress', 'N/A')
         parent_email = info.get('ParentEmailAddress', 'N/A')
         found_in_csv = True

     if not found_in_csv:
         print(f"  DEBUG: Could not find details for '{name}' in {STUDENT_CSV}. Using N/A.")

     duration_seen = recognition_duration.get(name, 0.0)
     duration_min = duration_seen / 60.0
     attendance_list.append({'StudentID': student_id, 'Name': name,
                             'StudentEmailAddress': student_email, # Changed column name
                             'ParentEmailAddress': parent_email,   # Added new column
                             'Status': status,
                             'DurationSeen_min': round(duration_min, 2)})

df_report = pd.DataFrame(attendance_list)
timestamp = time.strftime("%Y-%m-%d_%I-%M-%S%p")
excel_filename = os.path.join(REPORTS_DIR, f"attendance_report_{timestamp}.xlsx")
try:
    df_report.to_excel(excel_filename, index=False)
    print(f"Attendance report saved to {excel_filename}")
except Exception as e:
    print(f"Error saving Excel report: {e}")

# --- Send Emails to Absentees ---
print("\nSending emails to absentees...")
absentees = df_report[df_report['Status'] == 'Absent']

if not EMAIL_SENDER or "@" not in EMAIL_SENDER or not EMAIL_PASSWORD or EMAIL_PASSWORD == "your_app_password":
    print("Email sender or password not configured correctly. Skipping email notifications.")
    print("Please configure EMAIL_SENDER and EMAIL_PASSWORD (use Gmail App Password if needed).")
elif absentees.empty:
    print("No absentees found.")
else:
    sent_count_student = 0; sent_count_parent = 0
    error_count = 0; skipped_count = 0
    print(f"Attempting to send emails via {EMAIL_SERVER}:{EMAIL_PORT} as {EMAIL_SENDER}...")
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(EMAIL_SERVER, EMAIL_PORT, context=context, timeout=30) as server:
            print("Connecting to SMTP server...")
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            print("Logged into email server successfully.")
            for index, row in absentees.iterrows():
                student_email = row['StudentEmailAddress']
                parent_email = row['ParentEmailAddress']
                recipient_name = row['Name']
                current_time_str = time.strftime('%Y-%m-%d at %I:%M:%S %p')
                
                # --- Send to Student ---
                if pd.notna(student_email) and isinstance(student_email, str) and "@" in student_email:
                    subject_student = "Absence Notification - Attendance System"
                    body_student = f"Dear {recipient_name},\n\nYou were marked as absent by the automated attendance system for the session on {current_time_str}.\n\nPlease contact the instructor/administrator if you believe this is an error.\n\nRegards,\nAttendance System"
                    em_student = EmailMessage()
                    em_student['From'] = EMAIL_SENDER
                    em_student['To'] = student_email
                    em_student['Subject'] = subject_student
                    em_student.set_content(body_student)
                    try:
                        print(f"Sending email to student {recipient_name} ({student_email})...")
                        server.send_message(em_student)
                        print(f"  > Email to student sent successfully.")
                        sent_count_student += 1
                    except Exception as e_send_student:
                        print(f"  > Failed to send email to student {recipient_name}: {e_send_student}")
                        error_count += 1
                    time.sleep(0.5) # Small delay
                else:
                    print(f"Skipping email for student {recipient_name} (invalid or missing student email).")
                    skipped_count +=1

                # --- Send to Parent ---
                if pd.notna(parent_email) and isinstance(parent_email, str) and "@" in parent_email:
                    subject_parent = f"Absence Notification for {recipient_name} - Attendance System"
                    body_parent = f"Dear Parent/Guardian of {recipient_name},\n\nThis is to inform you that {recipient_name} was marked as absent by the automated attendance system for the class session on {current_time_str}.\n\nPlease contact the instructor/administrator for further details if needed.\n\nRegards,\nAttendance System"
                    em_parent = EmailMessage()
                    em_parent['From'] = EMAIL_SENDER
                    em_parent['To'] = parent_email
                    em_parent['Subject'] = subject_parent
                    em_parent.set_content(body_parent)
                    try:
                        print(f"Sending email to parent of {recipient_name} ({parent_email})...")
                        server.send_message(em_parent)
                        print(f"  > Email to parent sent successfully.")
                        sent_count_parent += 1
                    except Exception as e_send_parent:
                        print(f"  > Failed to send email to parent of {recipient_name}: {e_send_parent}")
                        error_count += 1
                    time.sleep(1.0) # Slightly longer delay before next student
                else:
                    print(f"Skipping email for parent of {recipient_name} (invalid or missing parent email).")
                    # No skipped_count increment here if student email was also skipped

    except smtplib.SMTPAuthenticationError:
        print("\nSMTP Authentication Error...") # (rest of error messages same as before)
        error_count = len(absentees)*2 - sent_count_student - sent_count_parent - skipped_count
    except smtplib.SMTPConnectError:
         print(f"\nSMTP Connection Error...")
         error_count = len(absentees)*2 - sent_count_student - sent_count_parent - skipped_count
    except ssl.SSLError as e:
         print(f"\nSSL Error during SMTP connection: {e}")
         error_count = len(absentees)*2 - sent_count_student - sent_count_parent - skipped_count
    except Exception as e:
        print(f"\nFailed to connect to email server or send emails: {e}")
        error_count = len(absentees)*2 - sent_count_student - sent_count_parent - skipped_count

    print(f"\nEmail Summary: Student Emails Sent={sent_count_student}, Parent Emails Sent={sent_count_parent}, Failed={error_count}, Skipped (due to missing address)={skipped_count}")

print("\nApplication finished.")

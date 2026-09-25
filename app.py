from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from openai import OpenAI
import PyPDF2
import re
import json
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
from io import BytesIO
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import auth, credentials, firestore
# ================== CẤU HÌNH & KHỞI TẠO ==================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ Không tìm thấy OPENROUTER_API_KEY trong biến môi trường!")

# Khởi tạo OpenAI client trỏ đến OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

GENERATION_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    raise ValueError("❌ Không tìm thấy FLASK_SECRET_KEY trong biến môi trường!")

# Cấu hình upload folder cho PDF
UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "phuongtay89@gmail.com").lower()
FIREBASE_COLLECTION = os.getenv("FIREBASE_USERS_COLLECTION", "taikhoan_hocsinh")

def initialize_firebase():
    if firebase_admin._apps:
        return firestore.client()

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")

    if service_account_json:
        cred = credentials.Certificate(json.loads(service_account_json))
    elif service_account_path:
        cred = credentials.Certificate(service_account_path)
    else:
        raise ValueError("Không tìm thấy FIREBASE_SERVICE_ACCOUNT_JSON hoặc FIREBASE_SERVICE_ACCOUNT_PATH trong biến môi trường!")

    firebase_admin.initialize_app(cred)
    return firestore.client()

firebase_db = initialize_firebase()

def get_firebase_web_config():
    return {
        "apiKey": os.getenv("FIREBASE_API_KEY", ""),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.getenv("FIREBASE_APP_ID", ""),
        "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", ""),
    }

def get_user_ref(uid):
    return firebase_db.collection(FIREBASE_COLLECTION).document(uid)

def score_from_level(level):
    return {
        "Giỏi": 90,
        "Khá": 75,
        "Đạt yêu cầu": 60,
        "Chưa đạt": 40,
    }.get(level, 60)

def split_history(history_text):
    return [line for line in (history_text or "").split("\n") if line.strip()]

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def normalize_student(uid, data):
    data = data or {}
    level = data.get("level", "Đạt yêu cầu")
    score = safe_int(data.get("score"), score_from_level(level))
    return {
        "id": uid,
        "uid": uid,
        "username": data.get("username") or data.get("email") or uid,
        "email": data.get("email", ""),
        "name": data.get("name") or data.get("display_name") or "Chưa đặt tên",
        "level": level,
        "score": score,
        "history": data.get("history", ""),
        "last_exchange": data.get("last_exchange", ""),
        "lydo": data.get("lydo", ""),
        "question_count": safe_int(data.get("question_count"), 0),
    }

def get_student(uid):
    doc = get_user_ref(uid).get()
    return normalize_student(uid, doc.to_dict() if doc.exists else {})

def save_student(uid, data):
    data["updated_at"] = firestore.SERVER_TIMESTAMP
    get_user_ref(uid).set(data, merge=True)

def list_students():
    students = []
    for doc in firebase_db.collection(FIREBASE_COLLECTION).stream():
        students.append(normalize_student(doc.id, doc.to_dict()))
    return sorted(students, key=lambda item: item["name"].lower())

def login_with_firebase_token(id_token, submitted_name=""):
    decoded_token = auth.verify_id_token(id_token)
    uid = decoded_token["uid"]
    email = (decoded_token.get("email") or "").lower()

    if email == ADMIN_EMAIL:
        session["admin_session"] = True
        session["admin_email"] = email
        return "admin"

    user_ref = get_user_ref(uid)
    user_doc = user_ref.get()
    current = normalize_student(uid, user_doc.to_dict() if user_doc.exists else {})
    stored_name = current["name"]
    display_name = (
        submitted_name
        or (stored_name if stored_name != "Chưa đặt tên" else "")
        or decoded_token.get("name")
        or email
        or uid
    ).strip()
    save_student(uid, {
        "uid": uid,
        "email": email,
        "username": email or uid,
        "name": display_name or current["name"],
        "display_name": decoded_token.get("name", ""),
        "photo_url": decoded_token.get("picture", ""),
        "level": current["level"],
        "score": current["score"],
        "history": current["history"],
        "last_exchange": current["last_exchange"],
        "lydo": current["lydo"],
        "question_count": current["question_count"],
        "auth_provider": "google",
        **({ "created_at": firestore.SERVER_TIMESTAMP } if not user_doc.exists else {}),
    })

    session["user_id"] = uid
    session["user_email"] = email
    return "student"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Biến toàn cục lưu nội dung tài liệu PDF
DOCUMENT_CONTEXT = {
    "text": "",
    "is_ready": False
}

# ================== ĐỌC TÀI LIỆU PDF ==================
def extract_pdf_text(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"⚠️ Lỗi khi đọc PDF {pdf_path}: {e}")
    return text

def load_all_documents(directory='./static'):
    """Đọc toàn bộ nội dung PDF và ghép thành một chuỗi văn bản."""
    all_text = ""
    if not os.path.exists(directory):
        print(f"Thư mục {directory} không tồn tại.")
        return ""
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    print(f"🔍 Tìm thấy {len(pdf_files)} tệp PDF trong {directory}...")
    for filename in pdf_files:
        pdf_path = os.path.join(directory, filename)
        content = extract_pdf_text(pdf_path)
        if content.strip():
            all_text += f"\n\n===== [Tài liệu: {filename}] =====\n{content}"
    total_chars = len(all_text)
    print(f"✅ Đã đọc tổng cộng {total_chars:,} ký tự từ {len(pdf_files)} tài liệu.")
    return all_text

def initialize_documents():
    """Tải toàn bộ tài liệu PDF vào bộ nhớ."""
    global DOCUMENT_CONTEXT
    print("⏳ Đang tải tài liệu...")
    doc_text = load_all_documents()
    if not doc_text.strip():
        print("⚠️ Không có tài liệu PDF nào để tải.")
        DOCUMENT_CONTEXT["text"] = ""  # Xóa sạch text cũ
        DOCUMENT_CONTEXT["is_ready"] = False
        return
    DOCUMENT_CONTEXT.update({
        "text": doc_text,
        "is_ready": True
    })
    print("🎉 Tải tài liệu hoàn tất!")

initialize_documents()

# ================== ĐÁNH GIÁ NĂNG LỰC ==================
def evaluate_student_level(history):
    recent_questions = "\n".join([msg for msg in history[-10:] if msg.startswith("👧 Học sinh:")])
    prompt = f"""
    Bạn là một **Giáo viên Toán THCS Song ngữ (Anh – Việt)**, có nhiệm vụ **đánh giá năng lực học tập và khả năng tự học của học sinh** dựa trên lịch sử câu hỏi gần đây.
    Dưới đây là **10 câu hỏi gần nhất của học sinh**:
    {recent_questions}

    ### 🎯 Yêu cầu:
    1. Đọc kỹ nội dung các câu hỏi, xác định:
    - Mức độ hiểu biết của học sinh về môn Toán.
    - Khả năng **diễn đạt logic**, **sử dụng thuật ngữ khoa học**, **tự tìm hiểu**.
    - Mức độ sử dụng **song ngữ Anh – Việt**: đúng, sai, hoặc thiếu tự nhiên.
    - Chỉ phân loại học sinh dựa trên các câu hỏi liên quan đến bài học hoặc nhận định về bộ môn toán.
    2. Phân loại năng lực học tập tổng quát thành **một trong 4 cấp độ**:
    **Yêu cầu chi tiết cho từng mức độ:**
    - **Giỏi**:
    - Câu hỏi nâng cao, kết hợp nhiều khái niệm.
    - Học sinh diễn đạt logic, giao tiếp bằng tiếng Anh học thuật, có khả năng phân tích và hỏi sâu hơn, câu hỏi thiên hướng tư duy tốt.
    - Nắm vững thuật ngữ khoa học, sử dụng tiếng Anh thành thạo.
    - **Khá**:
    - Câu hỏi hiểu khái niệm cơ bản, áp dụng trực tiếp nhưng vẫn có thử thách nhỏ.
    - Học sinh có thể sử dụng tiếng Anh tương đối, còn một vài lỗi nhỏ nhưng diễn đạt tốt.
    - **Đạt Yêu cầu**:
    - Câu hỏi cơ bản, tập trung kiểm tra khái niệm và kỹ năng tính toán.
    - Học sinh còn sai thuật ngữ, ít sử dụng tiếng Anh, câu hỏi có thể sai sót về mặt ngữ pháp tiếng anh.
    - **Chưa đạt**:
    - Câu hỏi rất cơ bản hoặc gợi nhớ khái niệm, không đòi hỏi tư duy cao.
    - Học sinh cần hỗ trợ thêm, ngôn ngữ đơn giản, chủ yếu tiếng Việt.
    4. Cần viết RÕ RÀNG, PHÂN TÍCH CHUYÊN SÂU.
    ### 📋 Định dạng đầu ra:
    Cấp độ: [Giỏi / Khá / Đạt yêu cầu / Chưa đạt]  
    Lý do: [Dựa trên dữ liệu 10 câu hỏi, hãy viết một đoạn nhận xét chuyên sâu gồm 3 ý: (1) Phân tích chủ đề toán học học sinh hay hỏi; (2) Đánh giá ưu điểm/khuyết điểm trong tư duy toán và từ vựng song ngữ; (3) Gợi ý định hướng giáo dục. KHÔNG viết quá ngắn, viết văn bản từ 100 - 200 từ, lời văn mạch lạc.]
    Ví dụ:
    Cấp độ: Khá
    Lý do: Học sinh thường xuyên quan tâm đến các khái niệm cơ bản nhưng đã bắt đầu mở rộng sang toán ứng dụng. Điểm mạnh là học sinh sử dụng tiếng Anh tương đối tốt, chỉ có vài lỗi ngữ pháp nhỏ, giao tiếp tự tin. Khuyết điểm nằm ở chỗ học sinh còn hay trình bày lúng túng khi gặp các hệ phương trình phức tạp hoặc số học lớn. Đề xuất giáo viên cung cấp thêm các bài tập về hệ thức Vi-ét, đồng thời sửa lỗi cấu trúc câu hỏi tiếng Anh cho hoàn thiện.
    """

    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_content = response.choices[0].message.content
        response_text = (raw_content or "").strip()
        # Extract level and reason from response
        level_match = re.search(r'Cấp độ: (Giỏi|Khá|Đạt yêu cầu|Chưa đạt)', response_text)
        lydo_match = re.search(r'Lý do:\s*(.+)', response_text, re.DOTALL)
        
        level = level_match.group(1) if level_match else "Đạt yêu cầu"
        lydo = lydo_match.group(1).strip() if lydo_match else "Không có lý do cụ thể."
        
        if level not in ['Giỏi', 'Khá', 'Đạt yêu cầu', 'Chưa đạt']:
            level = 'Đạt yêu cầu'
        return level, lydo
    except Exception as e:
        print(f"❌ Lỗi đánh giá: {e}")
        return 'Đạt yêu cầu', 'Đánh giá không thành công do lỗi hệ thống.'


# ================== ĐỊNH DẠNG TRẢ LỜI ==================
def format_response(response):
    # Bảo vệ cú pháp LaTeX bằng cách tạm thời thay thế
    latex_matches = []
    def store_latex(match):
        latex_matches.append(match.group(0))
        return f"__LATEX_{len(latex_matches)-1}__"
    
    # Thay thế các đoạn LaTeX nội dòng ($...$) và độc lập ($$...$$)
    response = re.sub(r'\$\$([^$]+)\$\$', store_latex, response)
    response = re.sub(r'\$([^$]+)\$', store_latex, response)

    # Headings
    response = re.sub(r'(?m)^###\s+(.*)', r'<h3 style="margin-top:12px; margin-bottom:8px; color:#2c3e50; font-size:16px;">\1</h3>', response)
    response = re.sub(r'(?m)^##\s+(.*)', r'<h2 style="margin-top:16px; margin-bottom:8px; color:#2c3e50; font-size:18px; border-bottom:1px solid #eee; padding-bottom:4px;">\1</h2>', response)
    response = re.sub(r'(?m)^#\s+(.*)', r'<h1 style="margin-top:20px; margin-bottom:12px; color:#1a252f; font-size:22px;">\1</h1>', response)

    # Bold & italic
    formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response)
    formatted = re.sub(r'(?<!\n)\*(?!\s)(.*?)(?<!\s)\*(?!\*)', r'<em>\1</em>', formatted)

    # Bullet lists
    formatted = re.sub(r'(?m)^\s*[\*\-]\s+(.*)', r'<div style="padding-left:16px;">• \1</div>', formatted)

    # Numbered lists
    formatted = re.sub(r'(?m)^(\d+)\.\s+(.*)', r'<div style="padding-left:16px;">\1. \2</div>', formatted)

    # Newlines → <br>, collapse 3+ br into 2
    formatted = formatted.replace('\n', '<br>')
    formatted = re.sub(r'(<br>\s*){3,}', '<br><br>', formatted)

    # Áp dụng highlight_terms cho các từ khóa toán học
    for term, color in highlight_terms.items():
        formatted = formatted.replace(term, f'<span style="background:{color};color:white;font-weight:600;padding:1px 6px;border-radius:4px;font-size:13px;">{term}</span>')

    # Khôi phục cú pháp LaTeX
    for i, latex in enumerate(latex_matches):
        formatted = formatted.replace(f"__LATEX_{i}__", latex)

    return formatted

# FORMAT TRẢ LỜI
highlight_terms = {
    # 🧮 TOÁN HỌC
    "Số tự nhiên": "#59C059",
    "Số nguyên": "#59C059",
    "Số hữu tỉ": "#59C059",
    "Số thập phân": "#59C059",
    "Phân số": "#59C059",
    "Tỉ số – Tỉ lệ": "#59C059",
    "Tỉ lệ thuận – Tỉ lệ nghịch": "#59C059",
    "Biểu thức đại số": "#59C059",
    "Hằng đẳng thức đáng nhớ": "#59C059",
    "Nhân, chia đa thức": "#59C059",
    "Phân tích đa thức thành nhân tử": "#59C059",
    "Căn bậc hai, căn bậc ba": "#59C059",
    "Lũy thừa – Căn thức": "#59C059",
    "Giải phương trình": "#59C059",
    "Phương trình bậc nhất một ẩn": "#59C059",
    "Hệ phương trình bậc nhất hai ẩn": "#59C059",
    "Bất phương trình": "#59C059",
    "Hàm số – Đồ thị hàm số": "#59C059",
    "Hàm số bậc nhất": "#59C059",
    "Tọa độ trong mặt phẳng": "#59C059",
    "Định lý Pythagoras": "#59C059",
    "Chu vi – Diện tích – Thể tích": "#59C059",
    "Tam giác": "#59C059",
    "Hình tròn – Hình cầu": "#59C059",
}


# ================== ROUTES ==================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        flash('Vui lòng đăng ký bằng Google để dữ liệu được lưu trên Firebase.', 'error')
        return redirect(url_for('register'))
    return render_template('register.html', firebase_config=get_firebase_web_config())

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        flash('Vui lòng đăng nhập bằng Google.', 'error')
        return redirect(url_for('login'))
    return render_template('login.html', firebase_config=get_firebase_web_config())

@app.route('/firebase-login', methods=['POST'])
def firebase_login():
    data = request.get_json(silent=True) or {}
    id_token = data.get('idToken')
    name = data.get('name', '').strip()
    if not id_token:
        return jsonify({'error': 'Thiếu Firebase ID token'}), 400

    try:
        role = login_with_firebase_token(id_token, name)
        if role == "admin":
            return jsonify({'redirect': url_for('admin')})
        return jsonify({'redirect': url_for('index')})
    except Exception as e:
        print(f"Firebase login error: {e}")
        return jsonify({'error': 'Không thể xác thực Google. Vui lòng thử lại.'}), 401

@app.route('/logout')
def logout():
    session.clear()
    flash('Đã đăng xuất thành công.', 'success')
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        flash('Vui lòng đăng nhập để tiếp tục.', 'error')
        return redirect(url_for('login'))
    rag_status = "✅ Đã tải tài liệu thành công" if DOCUMENT_CONTEXT["is_ready"] else "⚠️ Chưa tải được tài liệu."
    user = get_student(session['user_id'])
    return render_template('index.html', rag_status=rag_status, user_level=user['level'], user_score=user['score'])

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Vui lòng đăng nhập'}), 401

    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'response': format_response('Con hãy nhập câu hỏi nhé!')})

    user = get_student(session['user_id'])
    history = split_history(user['history'])
    history.append(f"👧 Học sinh: {user_message}")

    # Lấy toàn bộ nội dung tài liệu làm ngữ cảnh
    related_context = DOCUMENT_CONTEXT["text"] if DOCUMENT_CONTEXT["is_ready"] else "Không có tài liệu nào."

    last_exchange = user['last_exchange']
    student_level = user['level']
    student_score = user['score']

    prompt = f"""Bạn là Thầy giáo Toán Song ngữ (Việt - Anh) dành cho học sinh THCS.
Sản phẩm do: GV hướng dẫn Nguyễn Phương Tây, HS thực hiện Phạm Lê Thế Dân lớp 9A1 trường THCS Hoài Phú - chỉ đề cập tác giả khi được hỏi.

## Vai trò và Giọng điệu
- Xưng "thầy", gọi học sinh là "con", giọng thân thiện, khích lệ.
- KHÔNG đánh giá năng lực học sinh trong câu trả lời.
- CHỈ trả lời câu hỏi liên quan đến Toán THCS. Nếu lệch chủ đề, nhẹ nhàng nhắc con quay lại Toán.

## Ngữ cảnh hiện tại
- Tài liệu tham khảo:
{related_context}

- Hội thoại trước đó:
{last_exchange if last_exchange else '(Đây là câu hỏi đầu tiên.)'}

- Năng lực hiện tại của học sinh: {student_level}
- Điểm đánh giá hiện tại: {student_score}/100
- Câu hỏi mới của học sinh: {user_message}

## Cách trả lời (BẮT BUỘC trình bày xen kẽ Anh - Việt theo từng bước)
- Nêu khái niệm hoặc công thức liên quan ở đầu (nếu cần), kèm theo bản dịch tiếng Anh ngay bên dưới.
- Giải bài theo từng bước, bắt đầu mỗi bước bằng Heading Markdown 3: `### Bước 1`, `### Bước 2`...
- Trong mỗi bước, hãy giải thích bằng **Tiếng Việt** trước.
- Ngay bên dưới giải thích tiếng Việt của bước đó, xuống dòng và cung cấp bản dịch **Tiếng Anh**, bắt đầu bằng: `👉 **English:**`
- Công thức toán dùng LaTeX: `$...$` cho nội dòng, `$$...$$` cho toán riêng dòng.

**VÍ DỤ MẪU DÀNH CHO 1 BƯỚC:**
### Bước 1: Rút gọn phương trình
Chia cả hai vế cho $2$, ta có:
$$x + 2 = 5$$
👉 **English:** 
Divide both sides by $2$, we have:
$$x + 2 = 5$$

- Kết thúc toàn bộ bài giải bằng 1 bài tập tương tự (cũng xen kẽ Việt - Anh).

## Điều chỉnh theo năng lực "{student_level}"

### Nếu Giỏi:
- Độ sâu: Giải thích chuyên sâu, mở rộng liên hệ kiến thức nâng cao, nêu nhiều cách giải khác nhau.
- Bài tập: Cho bài nâng cao, có tính thử thách, yêu cầu tư duy sáng tạo.
- Tiếng Anh: Dùng từ vựng học thuật nâng cao (quadratic equation, derive the formula, consecutive integers).
- Khuyến khích: Khen khả năng tư duy, gợi mở hướng nghiên cứu thêm.

### Nếu Khá:
- Độ sâu: Giải thích chi tiết, có ví dụ minh hoạ cụ thể, nêu 1-2 cách giải.
- Bài tập: Bài ở mức khá, áp dụng trực tiếp nhưng có chút biến đổi.
- Tiếng Anh: Dùng từ vựng trung cấp, câu rõ ràng (solve the equation, find the value).
- Khuyến khích: Động viên con tiếp tục phát huy, nhấn mạnh điểm làm tốt.

### Nếu Đạt yêu cầu:
- Độ sâu: Giải thích từng bước nhỏ, CỰC KỲ chi tiết, không bỏ qua bước nào.
- Bài tập: Bài cơ bản, tính toán trực tiếp theo công thức, dễ áp dụng.
- Tiếng Anh: Dùng từ đơn giản, câu ngắn gọn (add, subtract, the answer is).
- Khuyến khích: Khen sự cố gắng, nhắc con không ngại hỏi lại nếu chưa hiểu.

### Nếu Chưa đạt:
- Độ sâu: Giải thích CỰC KỲ DỄ HIỂU, dùng ví dụ đời thường gần gũi, hình ảnh trực quan.
- Bài tập: Bài nhập môn, cầm tay chỉ việc, cho sẵn gợi ý từng bước.
- Tiếng Anh: Chỉ dịch ý chính, dùng từ rất cơ bản, kèm phiên âm nếu cần.
- Khuyến khích: Rất nhẹ nhàng, kiên nhẫn, nhấn mạnh "con làm được", khen từng bước nhỏ.
## Quy tắc trình bày
- Dùng **Markdown**: **in đậm**, *in nghiêng*, bullet list (`- ` hoặc `* `), numbered list (`1. `).
- Công thức dùng LaTeX `$...$` và `$$...$$`.
- Đối với các khái niệm khoa học, hãy trình bày rõ ràng, giải thích để học sinh thực sự hiểu chú không chỉ để trả lời.
- Ngắn gọn, súc tích — KHÔNG viết quá dài dòng.
- Nếu bài dài, chia thành **Phần 1**, **Phần 2** và hỏi: *"Con có muốn thầy tiếp tục không?"*
"""


    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = response.choices[0].message.content or "Thầy không thể trả lời lúc này, con thử lại nhé!"

        last_exchange = f"👧 Học sinh: {user_message}\n🧑‍🏫 Thầy/Cô: {ai_text}"

        # Đánh giá level nếu đủ 10 câu hỏi mới
        student_questions = [msg for msg in history if msg.startswith("👧 Học sinh:")]
        new_level = user['level']
        new_score = user['score']
        lydo = user['lydo']
        if len(student_questions) % 10 == 0:
            new_level, lydo = evaluate_student_level(history)
            new_score = score_from_level(new_level)
            user['level'] = new_level
            user['lydo'] = lydo
            user['score'] = new_score
            print(f"User {user['username']} level updated to {new_level}, score {new_score}, reason: {lydo}")

        # Lưu lịch sử, điểm số và đánh giá vào Firebase.
        save_student(session['user_id'], {
            'history': '\n'.join([msg.strip() for msg in history]),
            'last_exchange': last_exchange,
            'level': new_level,
            'score': new_score,
            'lydo': lydo,
            'question_count': len(student_questions),
        })

        return jsonify({
            'response': format_response(ai_text),
            'level': new_level,
            'score': new_score,
        })

    except Exception as e:
        print(f"❌ Lỗi AI: {e}")
        return jsonify({'response': format_response("Thầy AI hơi mệt, con thử lại sau nhé!")})
# QUẢN LÝ HỌC SINH
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'admin_session' not in session:
        flash(f'Vui lòng đăng nhập Google bằng email admin: {ADMIN_EMAIL}', 'error')
        return redirect(url_for('login'))
    
    # Xử lý upload file PDF
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            flash('Không có file được chọn.', 'error')
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash(f'Upload {filename} thành công! Đã cập nhật tài liệu.', 'success')
            initialize_documents()
        else:
            flash('Chỉ chấp nhận file PDF!', 'error')
    
    pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')] if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    
    # Lấy dữ liệu taikhoan_hocsinh + tên học sinh từ Firebase
    taikhoan_hocsinh = list_students()
    user_data = []
    for user in taikhoan_hocsinh:
        user_data.append({
            'id': user['id'],
            'username': user['username'],
            'name': user['name'] or "Chưa đặt tên",  # HIỂN THỊ TÊN
            'level': user['level'],
            'score': user['score'],
            'lydo': user['lydo'],
            'question_count': user['question_count'],
            'history': user['history'] if user['history'] else 'Chưa có lịch sử'
        })
    
    return render_template('admin.html', pdf_files=pdf_files, user_data=user_data)

@app.route('/admin/delete_pdf/<filename>', methods=['POST'])
def delete_pdf(filename):
    if 'admin_session' not in session or not session['admin_session']:
        flash('Bạn không có quyền truy cập.', 'error')
        return redirect(url_for('admin'))
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            flash(f'Xóa file {filename} thành công! Đã cập nhật tài liệu.', 'success')
            initialize_documents()  # Tải lại tài liệu sau khi xóa
        except Exception as e:
            flash(f'Lỗi khi xóa file {filename}: {str(e)}', 'error')
    else:
        flash(f'File {filename} không tồn tại.', 'error')
    
    return redirect(url_for('admin'))

@app.route('/admin/export_csv')
def export_csv():
    if 'admin_session' not in session or not session['admin_session']:
        flash('Bạn không có quyền truy cập.', 'error')
        return redirect(url_for('admin'))
    
    taikhoan_hocsinh = list_students()
    user_data = []
    for user in taikhoan_hocsinh:
        user_data.append({
            'ID': user['id'],
            'Tên đăng nhập': user['username'],
            'Tên học sinh': user['name'] or "Chưa đặt tên",  # THÊM CỘT TÊN
            'Năng lực': user['level'],
            'Điểm số': user['score'],
            'Số câu hỏi': user['question_count'],
            'Lý do': user['lydo'],
            'Lịch sử': user['history'] if user['history'] else 'Chưa có lịch sử'
        })
    
    df = pd.DataFrame(user_data)
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='ket_qua_hoc_tap.csv'
    )
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_session', None)
    flash('Đã đăng xuất admin.', 'success')
    return redirect(url_for('admin'))

# ================== CHẠY APP ==================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

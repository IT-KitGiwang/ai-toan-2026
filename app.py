from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
import PyPDF2
import re
from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.sql import text
import pandas as pd
from io import BytesIO
from werkzeug.utils import secure_filename
# ================== CẤU HÌNH & KHỞI TẠO ==================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ Không tìm thấy GEMINI_API_KEY trong biến môi trường!")

genai.configure(api_key=api_key)

GENERATION_MODEL = 'gemini-2.5-flash-lite'
EMBEDDING_MODEL = 'text-embedding-004'

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.config["SESSION_TYPE"] = "filesystem"
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)
Session(app)
# Cấu hình upload folder cho PDF
UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class User(db.Model):
    __tablename__ = 'taikhoan_hocsinh'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.Text, default='')
    level = db.Column(db.String(20), default='Đạt yêu cầu')
    history = db.Column(db.Text, default='')
    lydo = db.Column(db.Text, default='')

with app.app_context():
    # Đảm bảo schema public tồn tại
    db.session.execute(text('CREATE SCHEMA IF NOT EXISTS public;'))
    db.create_all()
    print("✅ Đã kiểm tra/tạo bảng taikhoan_hocsinh trong schema public")

# Biến toàn cục cho RAG
RAG_DATA = {
    "chunks": [],
    "embeddings": np.array([]),
    "is_ready": False
}

# ================== ĐỌC & CHIA CHUNKS ==================
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

def create_chunks_from_directory(directory='./static', chunk_size=400):
    all_chunks = []
    if not os.path.exists(directory):
        print(f"Thư mục {directory} không tồn tại.")
        return []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    print(f"🔍 Tìm thấy {len(pdf_files)} tệp PDF trong {directory}...")
    for filename in pdf_files:
        pdf_path = os.path.join(directory, filename)
        content = extract_pdf_text(pdf_path)
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size].strip()
            if chunk:
                all_chunks.append(f"[Nguồn: {filename}] {chunk}")
    print(f"✅ Đã tạo tổng cộng {len(all_chunks)} đoạn văn (chunks).")
    return all_chunks

def embed_with_retry(texts, model_name, max_retries=5):
    all_embeddings = []
    for text in texts:
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(model=model_name, content=text)
                all_embeddings.append(result["embedding"])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Thử lại lần {attempt+1}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    print(f"💥 Thất bại sau {max_retries} lần: {e}")
                    raise
    return np.array(all_embeddings)

def initialize_rag_data():
    global RAG_DATA
    print("⏳ Đang khởi tạo dữ liệu RAG...")
    chunks = create_chunks_from_directory()
    if not chunks:
        print("Không có dữ liệu để nhúng.")
        return
    try:
        embeddings = embed_with_retry(chunks, EMBEDDING_MODEL)
        RAG_DATA.update({
            "chunks": chunks,
            "embeddings": embeddings,
            "is_ready": True
        })
        print("🎉 Khởi tạo RAG hoàn tất!")
    except Exception as e:
        print(f"❌ KHÔNG THỂ KHỞI TẠO RAG: {e}")
        RAG_DATA["is_ready"] = False

initialize_rag_data()

# ================== TRUY XUẤT NGỮ CẢNH ==================
def retrieve_context(query, top_k=3):
    if not RAG_DATA["is_ready"]:
        return "Không có tài liệu RAG nào được tải."
    try:
        query_vec = embed_with_retry([query], EMBEDDING_MODEL)[0].reshape(1, -1)
        sims = cosine_similarity(query_vec, RAG_DATA["embeddings"])[0]
        top_idxs = np.argsort(sims)[-top_k:][::-1]
        return "\n\n---\n\n".join([RAG_DATA["chunks"][i] for i in top_idxs])
    except Exception as e:
        print(f"❌ Lỗi RAG: {e}")
        return "Lỗi khi tìm kiếm ngữ cảnh."

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
    4. Viết kết quả ngắn gọn, có lý do súc tích.
    ### 📋 Định dạng đầu ra:
    Cấp độ: [Giỏi / Khá / Đạt yêu cầu / Chưa đạt]  
    Lý do: [Giải thích lý do rõ ràng, phân tích định hướng cho giáo viên hỗ trợ, tối đa 150–200 từ.]
    Ví dụ:
    Cấp độ: Khá
    Lý do: Học sinh thường hỏi các câu về khái niệm cơ bản nhưng có kết hợp thêm một số bài toán ứng dụng nhỏ. 
    Học sinh sử dụng tiếng Anh tương đối tốt, chỉ có một vài lỗi ngữ pháp. 
    Câu hỏi thể hiện tư duy logic, khả năng tự tìm hiểu, nhưng vẫn cần hướng dẫn thêm để nâng cao kỹ năng.
    """

    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        response_text = response.text.strip()
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

    # Áp dụng định dạng Markdown
    formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong style="font-weight:700;">\1</strong>', response)
    formatted = re.sub(r'(?<!\n)\*(?!\s)(.*?)(?<!\s)\*(?!\*)', r'<em style="font-style:italic;">\1</em>', formatted)
    formatted = re.sub(r'(?m)^\s*\*\s+(.*)', r'• <span style="line-height:1.6;">\1</span>', formatted)
    formatted = formatted.replace('\n', '<br>')

    # Áp dụng highlight_terms cho các từ khóa toán học
    for term, color in highlight_terms.items():
        formatted = formatted.replace(term, f'<span style="line-height:1.6; background:{color}; color:white; font-weight:bold; padding:2px 4px; border-radius:4px;">{term}</span>')

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
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name', '').strip()  # LẤY TÊN HỌC SINH
        if not username or not password:
            flash('Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.', 'error')
            return redirect(url_for('register'))
        if not name:
            flash('Vui lòng nhập tên học sinh.', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Tên đăng nhập đã tồn tại.', 'error')
            return redirect(url_for('register'))

        try:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            user = User(username=username, password=hashed_password, name=name)
            db.session.add(user)
            db.session.commit()
            flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            print(f"Error during registration: {str(e)}")
            flash(f'Lỗi khi đăng ký: {str(e)}', 'error')
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.', 'error')
            return redirect(url_for('login'))
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['history'] = user.history.split('\n') if user.history else []
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('index'))
        flash('Tên đăng nhập hoặc mật khẩu không đúng.', 'error')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    if 'user_id' in session:
        user = db.session.get(User, session['user_id'])
        if user:
            user.history = '\n'.join(session.get('history', []))
            db.session.commit()
            print(f"User {user.username} history updated")
        else:
            print(f"User with ID {session['user_id']} not found")
    session.clear()
    flash('Đã đăng xuất thành công.', 'success')
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        flash('Vui lòng đăng nhập để tiếp tục.', 'error')
        return redirect(url_for('login'))
    rag_status = "✅ Đã tải tài liệu RAG thành công" if RAG_DATA["is_ready"] else "⚠️ Chưa tải được tài liệu RAG."
    user = db.session.get(User, session['user_id'])
    if not user:
        flash('Người dùng không tồn tại. Vui lòng đăng nhập lại.', 'error')
        return redirect(url_for('login'))
    return render_template('index.html', rag_status=rag_status, user_level=user.level)

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Vui lòng đăng nhập'}), 401

    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'response': format_response('Con hãy nhập câu hỏi nhé!')})

    # Load history from session
    history = session.get('history', [])
    history.append(f"👧 Học sinh: {user_message}")

    # 🔍 Truy xuất ngữ cảnh RAG
    related_context = retrieve_context(user_message)
    recent_history = "\n".join(history[-10:])

    # Lấy level từ DB
    user = db.session.get(User, session['user_id'])
    if not user:
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    student_level = user.level

    prompt = f"""
    Bạn là **Thầy giáo Song ngữ Việt – Anh**, chuyên dạy môn Toán THCS, do nhóm học sinh: 1) Hồ Mai Phương 2) Hoàng Nguyên Thanh Tuyền và giáo viên hướng dẫn: Lê Văn Rin tạo ra, không cần trả lời nhóm tác giả nếu không cần thiết.
    Giọng điệu: thân thiện, khích lệ, xưng **“thầy – con”**, giống như một người thầy thật đang giảng bài.
    Không đánh giá năng lực của học sinh trong câu trả lời.
    Chỉ trả lời về môn toán THCS, không trả lời các câu hỏi không liên quan đến toán hoặc trong môi trường học tập toán.
    ---
    ### 🧠 **Thông tin nền:**
    - 📚 **Tài liệu tham khảo:**  
    {related_context}
    - 💬 **Lịch sử hội thoại gần đây:**  
    {recent_history}
    - 👨‍🎓 **Năng lực hiện tại của học sinh:** {student_level}
    - ❓ **Câu hỏi mới:** {user_message}
    ---
    ### 🎯 **Nhiệm vụ của thầy:**
    1. **Hiểu rõ câu hỏi** — có thể bằng **tiếng Việt**, **tiếng Anh**, hoặc **cả hai**.  
    2. **Trả lời song ngữ** theo từng câu, từng đoạn:
    - Giải thích bằng **Tiếng Việt** trước theo từng câu, từng bước.
    - Sau đó viết phần dịch tương ứng, mở đầu bằng:  
        👉 <span style="line-height:1.6; background: darkblue; color:white; font-weight:bold; padding:2px 4px; border-radius:4px;">English Version</span>
    3. **Trình bày công thức, biểu thức khoa học bằng LaTeX**, sử dụng:  
    - `$...$` cho công thức trong dòng  
    - `$$...$$` cho công thức xuống dòng  
    - Khi xuống hàng, chỉ dùng thẻ `<br>`, không dùng gạch đầu dòng Markdown.
    Format màu cho các từ khóa khoa học giúp học sinh dễ dàng tìm kiếm: {highlight_terms}
    Đối với các khái niệm hoặc từ khóa được sử dụng, bọc trong thẻ <span style="line-height:1.6; background: (màu dựa trên highlight_terms); color:white; font-weight:bold; padding:2px 4px; border-radius:4px;">{{term}}</span>
    4. **Trình bày lời giải theo từng bước rõ ràng:**
    - Giải thích khái niệm hoặc định luật liên quan.  
    - Hướng dẫn cách giải nếu là bài tập.  
    - Cho **ví dụ tương tự** để luyện tập.  
    - Dịch các **thuật ngữ khoa học quan trọng** sang tiếng Anh học thuật tương ứng.  
    5. **Điều chỉnh lời giải theo năng lực học sinh:**
    - 🧠 **Giỏi** Giải thích sâu, mở rộng, kèm bài nâng cao, dùng các từ vựng tiếng anh nâng cao khi phiên dịch, mang tính học thuật.
    - 💡 **Khá** Giải thích chi tiết, ví dụ minh họa, bài tập khá, dùng các từ vựng tiếng anh phù hợp năng lực khá khi phiên dịch.
    - 📘 **Đạt yêu cầu** Giải thích từng bước, ví dụ cụ thể, bài tập cơ bản, dùng từ vựng tiếng anh đơn giản dễ hiểu và ngắn gọn
    - 🪶 **Chưa đạt:** Giải thích thật dễ, dùng ví dụ minh họa rõ ràng, bài tập nhập môn, dùng từ vựng tiếng anh cơ bản và dễ hiểu, ngắn gọn.
    6. **Nếu câu trả lời quá dài:**
    - Giữ ngữ cảnh liên tục giữa các phần.  
    - Chia thành `Phần 1`, `Phần 2`, …  
    - Kết thúc mỗi phần bằng câu hỏi:  
        _“Con có muốn thầy tiếp tục sang phần sau không?”_
    ---
    ### ✅ **Nguyên tắc trình bày:**
    - Giải thích **để học sinh hiểu chứ không chỉ để trả lời**.  
    - Duy trì giọng điệu tích cực, khuyến khích.  
    - Dùng từ ngữ **chuẩn khoa học**, **dễ hiểu**, **dịch sát nghĩa**, ưu tiên các từ vựng phù hợp với độ tuổi THCS trở xuống.  
    - Luôn dịch tiếng anh theo từng bước.
    - Luôn ưu tiên sự ngắn gọn, dễ hiểu, tránh lan man dài dòng.
    """


    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        ai_text = response.text

        # Lưu trả lời AI vào history
        history.append(f"🧑‍🏫 Thầy/Cô: {ai_text}")

        # Đánh giá level nếu đủ 5 câu hỏi mới
        student_questions = [msg for msg in history if msg.startswith("👧 Học sinh:")]
        if len(student_questions) % 10 == 0:
            new_level, lydo = evaluate_student_level(history)
            user.level = new_level
            user.lydo = lydo  # lưu lý do vào cột lydo
            db.session.commit()
            print(f"User {user.username} level updated to {new_level} with reason: {lydo}")

        # Lưu lịch sử câu hỏi học sinh vào session và database
        history_questions = student_questions
        # Đảm bảo mỗi tin nhắn xuống dòng riêng biệt
        session['history'] = history_questions
        user.history = '\n'.join([msg.strip() for msg in history_questions])  # Xóa khoảng trắng thừa và nối bằng \n
        db.session.commit()
        session.modified = True
        print(f"User {user.username} history updated in database: {user.history}")

        return jsonify({'response': format_response(ai_text)})

    except Exception as e:
        print(f"❌ Lỗi Gemini: {e}")
        return jsonify({'response': format_response("Thầy Gemini hơi mệt, con thử lại sau nhé!")})
# QUẢN LÝ HỌC SINH
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'admin_session' not in session:
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            if username == 'thaygiao1234':
                user = User.query.filter_by(username=username).first()
                if user and check_password_hash(user.password, password):
                    session['admin_session'] = True
                    flash('Đăng nhập admin thành công!', 'success')
                    return redirect(url_for('admin'))
                else:
                    flash('Tên đăng nhập hoặc mật khẩu không đúng.', 'error')
            else:
                flash('Tên đăng nhập admin không đúng.', 'error')
        return render_template('admin_login.html')
    
    # Xử lý upload file PDF
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            flash('Không có file được chọn.', 'error')
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash(f'Upload {filename} thành công! Đã cập nhật RAG.', 'success')
            initialize_rag_data()
        else:
            flash('Chỉ chấp nhận file PDF!', 'error')
    
    pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')] if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    
    # Lấy dữ liệu taikhoan_hocsinh + tên học sinh
    taikhoan_hocsinh = User.query.all()
    user_data = []
    for user in taikhoan_hocsinh:
        user_data.append({
            'id': user.id,
            'username': user.username,
            'name': user.name or "Chưa đặt tên",  # HIỂN THỊ TÊN
            'level': user.level,
            'lydo': user.lydo,
            'history': user.history if user.history else 'Chưa có lịch sử'
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
            flash(f'Xóa file {filename} thành công! Đã cập nhật RAG.', 'success')
            initialize_rag_data()  # Re-init RAG sau khi xóa
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
    
    taikhoan_hocsinh = User.query.all()
    user_data = []
    for user in taikhoan_hocsinh:
        user_data.append({
            'ID': user.id,
            'Tên đăng nhập': user.username,
            'Tên học sinh': user.name or "Chưa đặt tên",  # THÊM CỘT TÊN
            'Năng lực': user.level,
            'Lý do': user.lydo,
            'Lịch sử': user.history if user.history else 'Chưa có lịch sử'
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

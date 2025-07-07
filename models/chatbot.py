import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import pygame
from gtts import gTTS
import tempfile
import re
import random
import textwrap
import pickle

# Inisialisasi pygame mixer
pygame.mixer.init()

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Chatbot Kesehatan Mental", page_icon="💬", layout="wide")

# Custom CSS untuk tampilan 
st.markdown("""
<style>
    /* Warna background */
    .stApp {
        background-color: #111827;
        color: white;
    }
    
    /* Styling untuk judul utama */
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    /* Container hijau muda */
    .info-container {
        background-color: #e0f0e0;
        padding: 40px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    /* Judul hijau */
    .green-title {
        color: #2e7d32;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Caption hitam */
    .black-caption {
        color: black;
        font-size: 1rem;
        text-align: center;
    }
    
    /* Judul fitur */
    .feature-title {
        font-size: 1.8rem;
        margin-bottom: 30px;
        color: white;
    }
    
    /* Card fitur */
    .feature-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    /* Judul fitur dalam card */
    .feature-name {
        color: #2e7d32;
        font-size: 1.5rem;
        font-weight: bold;
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    
    /* Tombol hijau */
    .green-button {
        background-color: #2e7d32;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        text-align: center;
        margin: 10px;
        font-weight: bold;
        cursor: pointer;
    }
    
    /* Tombol kembali */
    .back-button {
        background-color: #f0f0f0;
        color: #333;
        padding: 10px 20px;
        border-radius: 20px;
        text-align: center;
        margin: 10px;
        font-weight: bold;
        cursor: pointer;
        border: 1px solid #ccc;
    }
    
    /* Untuk membuat chat container terlihat lebih baik */
    .stChatMessage {
        max-width: 100% !important;
        word-wrap: break-word !important;
    }
    .stChatMessage p {
        word-wrap: break-word !important;
        white-space: normal !important;
    }
    
    /* Untuk memastikan text area cukup besar */
    .stTextArea textarea {
        min-height: 150px !important;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk menghasilkan suara
def text_to_speech(text, language='id'):
    try:
        # Membuat file audio sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_audio.name)
            return temp_audio.name
    except Exception as e:
        st.error(f"Gagal menghasilkan suara: {e}")
        return None

# Fungsi untuk memutar suara
def play_audio(audio_file):
    try:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        st.error(f"Gagal memutar audio: {e}")

# Fungsi preprocessing pertanyaan
def preprocess_question(question):
    # Standarisasi pertanyaan
    question = question.lower().strip()
    # Hapus tanda baca berlebih
    question = re.sub(r'[?!.,;:]{2,}', '?', question)
    return question

# Ekstrak pasangan Q&A dari dokumen
def extract_qa_pairs(documents):
    qa_pairs = []
    for doc in documents:
        content = doc.page_content
        # Split berdasarkan "Q:" untuk mendapatkan pasangan Q&A
        parts = content.split("Q:")
        for part in parts[1:]:  # Skip bagian pertama kosong
            if "A:" in part:
                q_text = part.split("A:")[0].strip().lower()
                a_text = part.split("A:")[1].strip()
                if q_text and a_text:
                    # Pastikan jawaban lengkap dengan menambahkan titik jika perlu
                    if not a_text.endswith(('.', '!', '?')):
                        a_text += '.'
                    qa_pairs.append({"question": q_text, "answer": a_text})
    return qa_pairs

# Load Dokumen PDF 
def load_documents():
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        st.error("Tidak ada dokumen PDF yang ditemukan di folder 'data/'.")
        st.stop()
    return documents

# Split Dokumen dengan pendekatan yang lebih sesuai untuk format Q&A
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=300,
        separators=["Q:", "\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_documents(documents)
    if not text_chunks:
        st.error("Gagal membagi dokumen menjadi bagian kecil.")
        st.stop()
    return text_chunks

# Embedding dan FAISS
def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    try:
        vector_store = FAISS.from_documents(text_chunks, embeddings)
    except Exception as e:
        st.error(f"Gagal membuat FAISS Index: {e}")
        st.stop()
    return vector_store

# Inisialisasi Model dengan model yang lebih kecil (cukup untuk fallback)
def load_local_model():
    model_name = "facebook/opt-125m"  # Model kecil cukup karena utamanya menggunakan dataset
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        generator = pipeline(
            'text-generation', 
            model=model, 
            tokenizer=tokenizer, 
            max_length=512,  # Diperpanjang untuk jawaban yang lebih lengkap
            temperature=0.7,
            do_sample=True
        )
        
        llm = HuggingFacePipeline(pipeline=generator)
        return llm
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Tampilkan status loading
with st.spinner("Memuat model dan data..."):
    # Load atau Buat FAISS Index
    db_path = "faiss_index"
    qa_pairs_path = "qa_pairs.pkl"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Load atau buat Q&A pairs
    if os.path.exists(qa_pairs_path):
        try:
            with open(qa_pairs_path, 'rb') as f:
                qa_pairs = pickle.load(f)
        except Exception as e:
            st.error(f"Gagal memuat Q&A pairs: {e}")
            documents = load_documents()
            text_chunks = split_documents(documents)
            qa_pairs = extract_qa_pairs(documents)
            with open(qa_pairs_path, 'wb') as f:
                pickle.dump(qa_pairs, f)
    else:
        documents = load_documents()
        text_chunks = split_documents(documents)
        qa_pairs = extract_qa_pairs(documents)
        with open(qa_pairs_path, 'wb') as f:
            pickle.dump(qa_pairs, f)
    
    # Load atau buat FAISS index
    if os.path.exists(db_path):
        try:
            vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Gagal memuat FAISS Index: {e}")
            documents = load_documents()
            text_chunks = split_documents(documents)
            vector_store = create_vector_store(text_chunks)
            vector_store.save_local(db_path)
    else:
        documents = load_documents()
        text_chunks = split_documents(documents)
        vector_store = create_vector_store(text_chunks)
        vector_store.save_local(db_path)

    # Inisialisasi LLM (hanya untuk fallback)
    llm = load_local_model()

    # Setup Memory Chatbot
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup ConversationalRetrievalChain jika model berhasil dimuat
    if llm:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            max_tokens_limit=512  # Batas token yang lebih besar untuk jawaban panjang
        )
    else:
        qa_chain = None

# Fungsi untuk mencari kecocokan langsung dari dataset Q&A
def find_exact_match(query, qa_pairs, threshold=0.7):
    query = query.lower().strip()
    best_match = None
    max_score = 0
    
    for pair in qa_pairs:
        q_text = pair["question"].lower()
        # Cek apakah query adalah substring dari pertanyaan
        if query in q_text or q_text in query:
            # Hitung skor kesamaan sederhana
            shorter = min(len(query), len(q_text))
            longer = max(len(query), len(q_text))
            if shorter > 0:
                score = shorter / longer
                if score > max_score:
                    max_score = score
                    best_match = pair
    
    if max_score >= threshold:
        return best_match["answer"]
    return None

# Fungsi untuk mencari kecocokan fuzzy
def find_fuzzy_match(query, qa_pairs):
    query_words = set(query.lower().split())
    best_match = None
    max_overlaps = 0
    
    for pair in qa_pairs:
        q_text = pair["question"].lower()
        q_words = set(q_text.split())
        
        # Hitung overlap kata
        overlaps = len(query_words.intersection(q_words))
        if overlaps > max_overlaps:
            max_overlaps = overlaps
            best_match = pair
    
    # Jika setidaknya ada 1 kata yang cocok
    if max_overlaps > 0 and len(query_words) > 0:
        overlap_ratio = max_overlaps / len(query_words)
        if overlap_ratio >= 0.4:  # Threshold bisa disesuaikan
            return best_match["answer"]
    
    return None

# Fungsi untuk memastikan jawaban lengkap
def ensure_complete_answer(answer):
    # Jika jawaban terpotong di tengah kalimat
    if len(answer.split()) > 50 and not answer.strip().endswith(('.', '!', '?')):
        # Cari titik terakhir dalam jawaban
        last_period = answer.rfind('.')
        if last_period > 0:
            return answer[:last_period+1]
        else:
            return answer + "..."  # Tambahkan elipsis jika tidak ada titik sama sekali
    return answer

# Fungsi Chatbot dengan pendekatan bertingkat
def chatbot_response(user_input):
    # Preprocess pertanyaan
    processed_input = preprocess_question(user_input)
    
    # 1. Coba temukan kecocokan langsung dari dataset Q&A
    exact_match = find_exact_match(processed_input, qa_pairs)
    if exact_match:
        complete_answer = ensure_complete_answer(exact_match)
        st.session_state['history'].append((user_input, complete_answer))
        return complete_answer
    
    # 2. Coba temukan kecocokan fuzzy
    fuzzy_match = find_fuzzy_match(processed_input, qa_pairs)
    if fuzzy_match:
        complete_answer = ensure_complete_answer(fuzzy_match)
        st.session_state['history'].append((user_input, complete_answer))
        return complete_answer
    
    # 3. Cari dari vector store sebagai fallback
    try:
        docs = vector_store.similarity_search(processed_input, k=1)
        if docs:
            content = docs[0].page_content
            # Cek apakah isi dokumen memuat format Q&A
            if "Q:" in content and "A:" in content:
                qa_parts = content.split("Q:")
                for part in qa_parts:
                    if "A:" in part:
                        a_text = part.split("A:")[1].strip()
                        if a_text:
                            complete_answer = ensure_complete_answer(a_text)
                            st.session_state['history'].append((user_input, complete_answer))
                            return complete_answer
    except Exception as e:
        st.error(f"Error dalam pencarian vektor: {e}")
    
    # 4. Gunakan LLM jika tersedia dan tidak menemukan jawaban dari dataset
    if qa_chain:
        try:
            result = qa_chain({"question": processed_input, "chat_history": st.session_state['history']})
            answer = result["answer"]
            complete_answer = ensure_complete_answer(answer)
            st.session_state['history'].append((user_input, complete_answer))
            return complete_answer
        except Exception as e:
            st.error(f"Error dalam LLM: {e}")
    
    # 5. Fallback ke jawaban default
    fallback_responses = [
        "Maaf, saya tidak memahami pertanyaan Anda. Bisakah Anda mengajukan pertanyaan dengan cara yang berbeda?",
        "Saya tidak yakin bagaimana menjawab itu. Mari kita bicara tentang topik kesehatan mental yang lain.",
        "Pertanyaan Anda di luar cakupan pengetahuan saya. Apakah ada topik lain yang ingin Anda diskusikan?"
    ]
    response = random.choice(fallback_responses)
    st.session_state['history'].append((user_input, response))
    return response

# Inisialisasi Session State
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Halo. Ceritakan bagaimana perasaanmu hari ini?"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Halo! 👋"]
    if 'audio_files' not in st.session_state:
        st.session_state['audio_files'] = [None]
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'  # Default ke halaman home
    if 'prev_page' not in st.session_state:
        st.session_state['prev_page'] = None  # Untuk menyimpan halaman sebelumnya

initialize_session_state()

# Fungsi untuk menangani navigasi
def navigate_to(page):
    st.session_state['prev_page'] = st.session_state['page']
    st.session_state['page'] = page
    st.rerun()

def go_back():
    if st.session_state['prev_page']:
        current_page = st.session_state['page']
        st.session_state['page'] = st.session_state['prev_page']
        st.session_state['prev_page'] = current_page
        st.rerun()
    else:
        st.session_state['page'] = 'home'
        st.rerun()

# Tampilan header dengan ikon
st.markdown('<div class="main-title">🧠 Selamat Datang di Aplikasi Kesehatan Mental</div>', unsafe_allow_html=True)

# Tombol kembali di bagian atas
if st.session_state['page'] != 'home':
    if st.button("← Kembali", key="back_button_top"):
        go_back()

# Kondisional rendering berdasarkan halaman aktif
if st.session_state['page'] == 'home':
    # Container hijau muda dengan judul dan caption
    st.markdown('''
    <div class="info-container">
        <div class="green-title">Kesehatan Mental adalah Prioritas</div>
        <div class="black-caption">Aplikasi ini dirancang untuk membantu Anda dalam menjaga kesehatan mental Anda. Hubungi profesional kesehatan mental jika mengalami gejala serius.</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Judul Fitur Utama
    st.markdown('<div class="feature-title">Fitur Utama</div>', unsafe_allow_html=True)
    
    # Tampilkan 3 fitur dalam kolom
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="feature-card">
            <div class="feature-name">💬 Chatbot Cerdas</div>
            <p style="color: black; text-align: center;">Diskusikan masalah kesehatan mental dengan chatbot kami</p>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Mulai Chat", key="start_chat"):
            navigate_to('chat')
    
    with col2:
        st.markdown('''
        <div class="feature-card">
            <div class="feature-name">📚 Sumber Informasi</div>
            <p style="color: black; text-align: center;">Akses informasi kesehatan mental terpercaya</p>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Lihat Informasi", key="view_info"):
            navigate_to('info')
    
    with col3:
        st.markdown('''
        <div class="feature-card">
            <div class="feature-name">🛠️ Alat Bantu</div>
            <p style="color: black; text-align: center;">Gunakan alat bantu untuk mengelola kesehatan mental</p>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Coba Alat", key="try_tools"):
            navigate_to('tools')

elif st.session_state['page'] == 'chat':
    # Tampilan Chatbot
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Tulis pertanyaan Anda:", placeholder="Contoh: Bagaimana cara mengatasi stres?", key='input')
            submit_button = st.form_submit_button(label='Kirim')

        if submit_button and user_input:
            with st.spinner("Mencari jawaban untuk Anda..."):
                output = chatbot_response(user_input)
                
                # Buat file audio untuk jawaban
                audio_file = text_to_speech(output)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
                st.session_state['audio_files'].append(audio_file)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                # Gunakan lebar kolom yang lebih besar untuk jawaban chatbot
                col1, col2 = st.columns([0.85, 0.15])
                
                with col1:
                    # Tampilkan pesan pengguna
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    
                    # Tangani wrap teks untuk menghindari pemotongan
                    generated_text = st.session_state["generated"][i]
                    # Tampilkan pesan chatbot dengan container yang lebih lebar
                    message(generated_text, key=str(i), avatar_style="fun-emoji")
                
                with col2:
                    st.write("")  # Beri ruang vertikal
                    st.write("")  # Beri ruang vertikal tambahan
                    # Tombol audio
                    if i < len(st.session_state['audio_files']) and st.session_state['audio_files'][i]:
                        if st.button(f"🔊", key=f"audio_{i}"):
                            play_audio(st.session_state['audio_files'][i])

    # Navigasi dan tombol reset
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset Chat"):
            st.session_state['history'] = []
            st.session_state['generated'] = ["Halo. Ceritakan bagaimana perasaanmu hari ini?"]
            st.session_state['past'] = ["Halo! 👋"]
            st.session_state['audio_files'] = [None]
            st.rerun()
    with col2:
        if st.button("Kembali ke Beranda"):
            navigate_to('home')

elif st.session_state['page'] == 'info':
    st.markdown('<div class="feature-title">📚 Sumber Informasi Kesehatan Mental</div>', unsafe_allow_html=True)
    
    # Konten informasi
    st.markdown("""
    <div style="background-color: white; color: black; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #2e7d32;">Apa itu Kesehatan Mental?</h3>
        <p>Kesehatan mental mencakup kesejahteraan emosional, psikologis, dan sosial kita. Ini memengaruhi cara kita berpikir, merasakan, dan bertindak.</p>
    </div>
    
    <div style="background-color: white; color: black; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #2e7d32;">Gejala Gangguan Mental</h3>
        <ul>
            <li>Perasaan sedih atau down yang berkepanjangan</li>
            <li>Kebingungan berpikir atau berkurangnya kemampuan berkonsentrasi</li>
            <li>Rasa takut atau khawatir yang berlebihan</li>
            <li>Perubahan mood yang ekstrem</li>
            <li>Menarik diri dari aktivitas sosial</li>
        </ul>
    </div>
    
    <div style="background-color: white; color: black; padding: 20px; border-radius: 10px;">
        <h3 style="color: #2e7d32;">Sumber Bantuan</h3>
        <p>Jika Anda atau orang terdekat mengalami gejala gangguan mental, segera hubungi:</p>
        <ul>
            <li>Psikolog atau psikiater terdekat</li>
            <li>Layanan kesehatan mental di rumah sakit</li>
            <li>Hotline kesehatan mental: 119</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Kembali ke Beranda"):
        navigate_to('home')

elif st.session_state['page'] == 'tools':
    st.markdown('<div class="feature-title">🛠️ Alat Bantu Kesehatan Mental</div>', unsafe_allow_html=True)
    
    # Konten alat bantu
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: white; color: black; padding: 20px; border-radius: 10px; height: 200px; margin-bottom: 20px;">
            <h3 style="color: #2e7d32;">Pengecekan Stres</h3>
            <p>Alat sederhana untuk mengecek tingkat stres Anda berdasarkan gejala yang dialami.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Gunakan Alat Stres", key="stress_tool"):
            navigate_to('stress_tool')
    
    with col2:
        st.markdown("""
        <div style="background-color: white; color: black; padding: 20px; border-radius: 10px; height: 200px; margin-bottom: 20px;">
            <h3 style="color: #2e7d32;">Jurnal Mood Harian</h3>
            <p>Catat perasaan Anda setiap hari untuk melacak perubahan mood dan pola emosi.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Gunakan Jurnal Mood", key="mood_journal"):
            navigate_to('mood_journal')
    
    if st.button("Kembali ke Beranda"):
        navigate_to('home')

elif st.session_state['page'] == 'stress_tool':
    st.markdown('<div class="feature-title">🧐 Pengecekan Tingkat Stres</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: white; color: black; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <p>Jawab pertanyaan berikut untuk mengecek tingkat stres Anda:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pertanyaan pengecekan stres
    questions = [
        "Seberapa sering Anda merasa gelisah atau cemas dalam seminggu terakhir?",
        "Seberapa sering Anda mengalami kesulitan tidur dalam seminggu terakhir?",
        "Seberapa sering Anda merasa kewalahan dengan tanggung jawab Anda?",
        "Seberapa sering Anda mengalami sakit kepala atau ketegangan otot tanpa alasan fisik yang jelas?",
        "Seberapa sering Anda merasa tidak bisa bersantai atau tenang?"
    ]
    
    answers = []
    for i, question in enumerate(questions):
        answers.append(st.select_slider(
            question,
            options=["Tidak pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"],
            key=f"stress_q_{i}"
        ))
    
    if st.button("Hitung Tingkat Stres", key="calculate_stress"):
        # Hitung skor stres
        score_map = {"Tidak pernah": 0, "Jarang": 1, "Kadang-kadang": 2, "Sering": 3, "Selalu": 4}
        total_score = sum(score_map[answer] for answer in answers)
        
        # Tentukan hasil
        if total_score <= 5:
            result = "Tingkat stres Anda rendah. Pertahankan gaya hidup sehat!"
            color = "green"
        elif total_score <= 10:
            result = "Tingkat stres Anda sedang. Coba lakukan relaksasi lebih sering."
            color = "orange"
        else:
            result = "Tingkat stres Anda tinggi. Pertimbangkan untuk berkonsultasi dengan profesional."
            color = "red"
        
        st.markdown(f"""
        <div style="background-color: white; color: black; padding: 20px; border-radius: 10px; margin-top: 20px; border-left: 5px solid {color}">
            <h3 style="color: {color};">Hasil Pengecekan Stres</h3>
            <p>Skor Anda: {total_score}/20</p>
            <p><strong>{result}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("Kembali ke Alat Bantu"):
        navigate_to('tools')

elif st.session_state['page'] == 'mood_journal':
    st.markdown('<div class="feature-title">📝 Jurnal Mood Harian</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: white; color: black; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <p>Catat perasaan Anda hari ini untuk melacak perubahan mood dari waktu ke waktu.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Form input jurnal
    date = st.date_input("Tanggal")
    mood = st.select_slider("Mood hari ini", options=["😭 Sangat Buruk", "😔 Buruk", "😐 Biasa saja", "🙂 Baik", "😊 Sangat Baik"])
    activities = st.text_area("Aktivitas hari ini")
    thoughts = st.text_area("Pikiran atau perasaan yang ingin dicatat")
    
    if st.button("Simpan Jurnal", key="save_journal"):
        if 'journals' not in st.session_state:
            st.session_state['journals'] = []
        
        st.session_state['journals'].append({
            'date': date,
            'mood': mood,
            'activities': activities,
            'thoughts': thoughts
        })
        
        st.success("Jurnal berhasil disimpan!")
    
    # Tampilkan riwayat jurnal jika ada
    if 'journals' in st.session_state and st.session_state['journals']:
        st.markdown("""
        <div style="margin-top: 30px;">
            <h3 style="color: white;">Riwayat Jurnal</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for journal in reversed(st.session_state['journals']):
            st.markdown(f"""
            <div style="background-color: white; color: black; padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                <h4>{journal['date']} - {journal['mood']}</h4>
                <p><strong>Aktivitas:</strong> {journal['activities']}</p>
                <p><strong>Pikiran/Perasaan:</strong> {journal['thoughts']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("Kembali ke Alat Bantu"):
        navigate_to('tools')
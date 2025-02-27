import customtkinter as ctk
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont,UnidentifiedImageError
from googletrans import Translator
import eng_to_ipa as ipa
from nltk.corpus import wordnet
import threading
import requests
from io import BytesIO
from duckduckgo_search import DDGS
def download_image_duckduckgo(query):
    save_dir = "flashcards"
    os.makedirs(save_dir, exist_ok=True)

    with DDGS() as ddgs:
        results = list(ddgs.images(query, max_results=10))  # Lấy 10 kết quả để tránh lỗi

    for i, result in enumerate(results):
        img_url = result.get("image", "")
        if not img_url:
            continue  # Bỏ qua nếu URL rỗng

        try:
            response = requests.get(img_url, stream=True, timeout=10)
            response.raise_for_status()

            # Kiểm tra dữ liệu có phải là ảnh không
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                print(f"⚠️ Bỏ qua ảnh {i+1}: Không phải file ảnh ({content_type})")
                continue

            # Đọc & lưu ảnh
            image = Image.open(BytesIO(response.content))
            filename = os.path.join(save_dir, f"{query}.jpg")
            if image.mode != "RGB":  # Nếu ảnh không phải RGB, chuyển đổi trước khi lưu
                image = image.convert("RGB")

            image.save(filename, "JPEG")

            print(f"✅ Đã lưu: {filename}")
            return filename  # Trả về đường dẫn ảnh ngay khi tải thành công

        except UnidentifiedImageError:
            print(f"❌ Lỗi ảnh {i+1}: Không nhận diện được ảnh.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Lỗi ảnh {i+1}: {e}")

    print("❌ Không tìm thấy ảnh hợp lệ!")
    return None  # Trả về None nếu không tải được ảnh nào
def load_word_list(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sorted(set(f.read().splitlines()))

word_list = load_word_list("The_Oxford_3000.txt")

def get_word_type(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return "Unknown"
    pos = synsets[0].pos()
    pos_map = {"n": "Danh từ", "v": "Động từ", "a": "Tính từ", "r": "Trạng từ"}
    return pos_map.get(pos, "Khác")

def translate_word(word):
    translator = Translator()
    result = translator.translate(word, src="en", dest="vi")
    pronunciation = ipa.convert(word)
    return result.text, pronunciation

def create_flashcard(word, progress_step):
    global flashcard_path
    
    keyword = word
    word_type = get_word_type(keyword)
    define_vn, pronunciation = translate_word(keyword)

    save_dir = "flashcards"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{keyword}.jpg"
    image_path = os.path.join(save_dir, filename)

    label_status.configure(text=f"Đang tạo: {keyword}")
    app.update_idletasks()

    # Tải ảnh từ DuckDuckGo
    image_path = download_image_duckduckgo(keyword)
    if image_path:
        img_pil = Image.open(image_path)  # Mở ảnh từ đường dẫn
    else:
        img_pil = Image.new("RGB", (500, 500), "white")  # Nếu không có ảnh, tạo nền trắng

    if img_pil is None:
        print(f"LỖI: Không tìm thấy ảnh cho từ '{keyword}', sử dụng nền trắng.")
        img_pil = Image.new("RGB", (500, 500), "white")

    # Resize ảnh về 500x500
    img_pil = img_pil.resize((500, 500))
    
    # Convert sang OpenCV để xử lý màu
    img_cv = np.array(img_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    progress_bar.set(0.5)
    app.update_idletasks()

    # Trích xuất màu chủ đạo bằng K-Means
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    pixels_hsv = img_hsv.reshape((-1, 3))
    pixels_hsv = np.float32(pixels_hsv)

    k = 3  # Số cụm màu
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(pixels_hsv, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Chuyển đổi màu chủ đạo sang RGB
    palette_rgb = [tuple(map(int, cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_HSV2RGB)[0][0])) for c in palette]

    progress_bar.set(0.7)
    app.update_idletasks()

    # Chọn màu chủ đạo phù hợp
    valid_colors = []
    for i, (h, s, v) in enumerate(palette):
        if v > 60 and s > 50:
            if not (h < 20 and s < 100 and v < 150):  # Loại bỏ màu quá tối hoặc xỉn
                valid_colors.append((labels.flatten().tolist().count(i), palette_rgb[i]))

    chosen_colors = sorted(valid_colors, reverse=True)  # Chọn màu xuất hiện nhiều nhất
    dominant_color = chosen_colors[0][1] if chosen_colors else (0, 0, 255)

    # Tạo flashcard
    card_width, card_height = 600, 800
    flashcard = Image.new("RGB", (card_width, card_height), "white")
    draw = ImageDraw.Draw(flashcard)

    # Viền màu chủ đạo
    border_width = 6
    draw.rectangle([(border_width, border_width), (card_width - border_width, card_height - border_width)],
                   outline=dominant_color, width=border_width)
    
    flashcard.paste(img_pil, ((card_width - 500) // 2, 20))

    # Font chữ
    font_path = "arial.ttf"
    font_en = ImageFont.truetype(font_path, 45)
    font_vi = ImageFont.truetype(font_path, 42)
    font_type = ImageFont.truetype(font_path, 40)

    # Thêm text lên flashcard
    draw.text(((card_width - draw.textlength(keyword.capitalize(), font=font_en)) // 2, 530),
              keyword.capitalize(), fill=dominant_color, font=font_en)
    draw.text(((card_width - draw.textlength(f"/{pronunciation}/", font=font_vi)) // 2, 590),
              f"/{pronunciation}/", fill="black", font=font_vi)
    draw.text(((card_width - draw.textlength(f"({word_type})", font=font_type)) // 2, 650),
              f"({word_type})", fill="gray", font=font_type)
    draw.text(((card_width - draw.textlength(define_vn.capitalize(), font=font_vi)) // 2, 710),
              define_vn.capitalize(), fill="black", font=font_vi)

    # Lưu flashcard
    flashcard = flashcard.convert("RGB")  # Chuyển từ RGBA sang RGB
    flashcard.save(image_path, "JPEG")


    progress_bar.set(progress_step)
    app.update_idletasks()
def list_flashcard():
    def run():
        total_words = len(word_list)
        if total_words == 0:
            label_status.configure(text="Không có từ nào trong danh sách!")
            return
        
        for i, word in enumerate(word_list):
            progress_step = (i + 1) / total_words  # Tiến trình từ 0 đến 1
            create_flashcard(word, progress_step)
        
        label_status.configure(text="Đã tạo xong tất cả flashcards!")
        progress_bar.set(1.0)
# Chạy tiến trình riêng để không chặn giao diện chính
    thread = threading.Thread(target=run)
    thread.daemon = True  # Đảm bảo chương trình tắt sẽ dừng thread
    thread.start()
# Giao diện
app = ctk.CTk()
app.iconbitmap("icon.ico")
app.title("Flashcard Lite")
app.geometry("400x450")
app.resizable(False, False)

progress_bar = ctk.CTkProgressBar(app, width=250)
progress_bar.pack(pady=5)
progress_bar.set(0)

label_status = ctk.CTkLabel(app, text="Nhấn nút để tạo flashcards", fg_color="transparent")
label_status.pack(pady=5)

btn_random = ctk.CTkButton(app, text="Tạo List Flashcard", command=list_flashcard, fg_color="orange")
btn_random.pack(pady=5)

app.mainloop()

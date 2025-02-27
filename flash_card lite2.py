import customtkinter as ctk
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from icrawler.builtin import GoogleImageCrawler
from googletrans import Translator
import eng_to_ipa as ipa
from nltk.corpus import wordnet
import threading
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

    label_status.configure(text=f"Đang tạo: {keyword}")
    app.update_idletasks()

    crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=keyword, max_num=1)

    file_list = os.listdir(save_dir)
    if file_list:
        old_file = os.path.join(save_dir, file_list[0])
        new_file = os.path.join(save_dir, filename)

        # Đọc ảnh và resize
        img_cv = cv2.imread(old_file)
        if img_cv is None:
            print(f"LỖI: Không thể đọc ảnh cho từ '{keyword}', bỏ qua từ này.")
            return  # Bỏ qua từ này nếu ảnh không hợp lệ
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        size_img = 500
        img_cv = cv2.resize(img_cv, (size_img, size_img))

        progress_bar.set(0.5)
        app.update_idletasks()

        # Trích xuất màu chủ đạo bằng K-means clustering
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
        pixels_hsv = img_hsv.reshape((-1, 3))
        pixels_hsv = np.float32(pixels_hsv)

        k = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, palette = cv2.kmeans(pixels_hsv, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        palette_rgb = [tuple(map(int, cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_HSV2RGB)[0][0])) for c in palette]

        progress_bar.set(0.7)
        app.update_idletasks()

        # Lọc màu
        valid_colors = []
        for i, (h, s, v) in enumerate(palette):
            if v > 60 and s > 50:
                if not (h < 20 and s < 100 and v < 150):
                    valid_colors.append((labels.flatten().tolist().count(i), palette_rgb[i]))

        rainbow_colors = []
        for count, color in valid_colors:
            h, s, v = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
            if 0 <= h <= 300:
                rainbow_colors.append((count, color))

        chosen_colors = rainbow_colors if rainbow_colors else valid_colors
        dominant_color = chosen_colors[0][1] if chosen_colors else (0, 0, 255)

        # Tạo flashcard
        img_pil = Image.fromarray(img_cv)
        card_width, card_height = 600, 800
        flashcard = Image.new("RGB", (card_width, card_height), "white")
        draw = ImageDraw.Draw(flashcard)

        # Vẽ viền màu chủ đạo
        border_width = 6
        draw.rectangle(
            [(border_width, border_width), (card_width - border_width, card_height - border_width)], 
            outline=dominant_color, width=border_width
        )
        flashcard.paste(img_pil, ((card_width - size_img) // 2, 20))

        # Chữ trên flashcard
        font = ImageFont.truetype("arial.ttf", 40)
        font_en = ImageFont.truetype("arial.ttf", 45)
        font_vi = ImageFont.truetype("arial.ttf", 42)

        draw.text(((card_width - draw.textlength(keyword.capitalize(), font=font)) // 2, size_img+50),
                  keyword.capitalize(), fill=dominant_color, font=font_en)
        draw.text(((card_width - draw.textlength(f"/{pronunciation}/", font=font)) // 2, size_img+110),
                  f"/{pronunciation}/", fill="black", font=font_vi)
        draw.text(((card_width - draw.textlength(f"({word_type})", font=font)) // 2,size_img+180),
                  f"({word_type})", fill="gray", font=font)
        draw.text(((card_width - draw.textlength(define_vn.capitalize(), font=font)) // 2,size_img+240),
                  define_vn.capitalize(), fill="black", font=font)

        flashcard.save(new_file)
        os.remove(old_file)
        flashcard_path = new_file

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

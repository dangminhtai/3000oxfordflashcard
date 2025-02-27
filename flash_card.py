import customtkinter as ctk
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from icrawler.builtin import GoogleImageCrawler
from googletrans import Translator
import eng_to_ipa as ipa
from nltk.corpus import wordnet
from threading import Thread
import webbrowser
from random import choice
import requests
import re
import threading

def load_word_list(file_path):
    with open(file_path, "r") as f:
        return sorted(set(f.read().splitlines()))

word_rd = load_word_list("google_word_eng.txt")
flashcard_path = None  # Biến lưu đường dẫn ảnh flashcard
def random_word():
    """ Chọn một từ ngẫu nhiên từ danh sách từ tiếng Anh """
    random_w = choice(word_rd)
    entry_keyword.delete(0, "end")
    entry_keyword.insert(0, random_w)

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

def create_flashcard():
    global flashcard_path
    keyword = entry_keyword.get()
    word_type = entry_type.get() or get_word_type(keyword)
    define_vn = entry_meaning.get()
    subnet_dir = entry_folder.get()
    progress_bar.set(0.1)

    if not define_vn:
        define_vn, pronunciation = translate_word(keyword)
    else:
        _, pronunciation = translate_word(keyword)
    
    save_dir = "flashcards"
    if subnet_dir:
        save_dir = os.path.join(save_dir, subnet_dir)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{keyword}.jpg"
    progress_bar.set(0.2)
    
    crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=keyword, max_num=1)
    progress_bar.set(0.4)
    
    file_list = os.listdir(save_dir)
    if file_list:
        old_file = os.path.join(save_dir, file_list[0])
        new_file = os.path.join(save_dir, filename)
        img_cv = cv2.imread(old_file)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        size_img=500
        img_cv = cv2.resize(img_cv, (size_img, size_img))
        progress_bar.set(0.5)
        
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
        pixels_hsv = img_hsv.reshape((-1, 3))
        pixels_hsv = np.float32(pixels_hsv)
        
        k = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, palette = cv2.kmeans(pixels_hsv, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        palette_rgb = [tuple(map(int, cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_HSV2RGB)[0][0])) for c in palette]
        progress_bar.set(0.7)
        
        valid_colors = []
        for i, (h, s, v) in enumerate(palette):
            if v > 60 and s > 50:  # Loại màu tối & màu nhạt
                if not (h < 20 and s < 100 and v < 150):  # Loại màu nâu
                    valid_colors.append((labels.flatten().tolist().count(i), palette_rgb[i]))

        rainbow_colors = []
        for count, color in valid_colors:
            h, s, v = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
            if 0 <= h <= 300:
                rainbow_colors.append((count, color))

        chosen_colors = rainbow_colors if rainbow_colors else valid_colors
        dominant_color = chosen_colors[0][1] if chosen_colors else (0, 0, 255)
        
        img_pil = Image.fromarray(img_cv)
        card_width, card_height = 600, 800
        flashcard = Image.new("RGB", (card_width, card_height), "white")
        draw = ImageDraw.Draw(flashcard)
        border_width = 6
        draw.rectangle(
            [(border_width, border_width), (card_width - border_width, card_height - border_width)], 
            outline=dominant_color, width=border_width
        )
        flashcard.paste(img_pil, ((card_width - size_img) // 2, 20))
        
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
        flashcard_path = new_file  # Lưu đường dẫn ảnh flashcard
        progress_bar.set(1.0)
        status_label.configure(text=f"🎴 Flashcard đã được tạo: {save_dir}/{filename}")
        btn_view.configure(state="normal")  # Bật nút xem ảnh

def view_flashcard():
    """ Mở ảnh flashcard trong trình xem ảnh mặc định """
    if flashcard_path and os.path.exists(flashcard_path):
        webbrowser.open(flashcard_path)
def get_suggestions(query):
    """Lấy gợi ý từ Google Suggest API (lọc chỉ từ tiếng Anh)"""
    url = f"https://suggestqueries.google.com/complete/search?client=firefox&q={query}"
    response = requests.get(url)
    if response.status_code == 200:
        suggestions = response.json()[1]
        english_suggestions = [s for s in suggestions if re.match(r"^[a-zA-Z\s]+$", s)]
        return english_suggestions
    return []

def update_suggestions():
    """Hàm chạy trên luồng riêng để lấy gợi ý mượt hơn"""
    global last_query
    query = entry_keyword.get().strip()
    if query and query != last_query:
        last_query = query  # Lưu lại từ khóa cuối cùng
        suggestions = get_suggestions(query)
        if suggestions:
            suggestion_label.configure(text=suggestions[0])  # Hiển thị từ gần nhất
        else:
            suggestion_label.configure(text="")

def on_key_release(event):
    """Gọi luồng riêng để tránh giật lag"""
    app.after(150, lambda: threading.Thread(target=update_suggestions).start())

def on_tab(event):
    """Hoàn thành từ khi nhấn Tab"""
    suggested_word = suggestion_label.cget("text")
    if suggested_word:
        entry_keyword.delete(0, "end")
        entry_keyword.insert(0, suggested_word)
        suggestion_label.configure(text="")  # Ẩn gợi ý sau khi chọn
    return "break"  # Ngăn Tab di chuyển focus
app = ctk.CTk()
app.iconbitmap("icon.ico") 
app.title("Word to Flashcard")
app.geometry("400x450")
italic_font = ("Arial", 14, "italic")

ctk.CTkLabel(app, text="Nhập từ tiếng Anh (yêu cầu kết nối Internet)",anchor="w", justify="left", width=250).pack()
entry_keyword = ctk.CTkEntry(app, placeholder_text="Enter text")
entry_keyword.pack(anchor="n")
entry_keyword.bind("<KeyRelease>", on_key_release)
entry_keyword.bind("<Tab>", on_tab)

suggestion_label = ctk.CTkLabel(app, text="", font=("Arial", 12, "italic"), text_color="gray")
suggestion_label.pack()

ctk.CTkLabel(app, text="Nhập loại từ (tính, danh, động, trạng,...)",anchor="w", justify="left", width=250).pack()
entry_type = ctk.CTkEntry(app, placeholder_text="Optional", placeholder_text_color="#A0A0A0", font=italic_font)
entry_type.pack(anchor="n")

ctk.CTkLabel(app, text="Nhập nghĩa tiếng Việt mà bạn biết",anchor="w", justify="left", width=250).pack()
entry_meaning = ctk.CTkEntry(app, placeholder_text="Optional", placeholder_text_color="#A0A0A0", font=italic_font)
entry_meaning.pack(anchor="n")

ctk.CTkLabel(app, text="Nhập thư mục chủ đề bạn muốn tạo",anchor="w", justify="left", width=250).pack()
entry_folder = ctk.CTkEntry(app, placeholder_text="Optional", placeholder_text_color="#A0A0A0", font=italic_font)
entry_folder.pack(anchor="n")

progress_bar = ctk.CTkProgressBar(app, width=250)
progress_bar.pack(pady=5)  # Căn sang trái bằng padding
progress_bar.set(0)

status_label = ctk.CTkLabel(app, text="")
status_label.pack()

btn_create = ctk.CTkButton(app, text="Tạo Flashcard", command=lambda: Thread(target=create_flashcard).start(),fg_color="red")
btn_create.pack()
btn_random = ctk.CTkButton(app, text="Random word", command=random_word, fg_color="orange",)
btn_random.pack(pady=5)
btn_view = ctk.CTkButton(app, text="Xem Flashcard", command=view_flashcard, state="disabled",fg_color="green")
btn_view.pack(pady=5)

last_query = ""
app.mainloop()

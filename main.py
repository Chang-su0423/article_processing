import tkinter as tk
from tkinter import filedialog, messagebox
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 下载nltk需要的数据
nltk.download('punkt')


def read_document(file_path):
    """读取输入文件中的内容"""
    with open(file_path, 'r') as file:
        document = file.read()
    return document


def preprocess_text(text):
    """将文档分割成句子"""
    sentences = nltk.sent_tokenize(text)
    return sentences


def compute_tfidf(sentences):
    """计算句子的TF-IDF特征矩阵"""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    return X, vectorizer


def perform_clustering(X, n_clusters):
    """使用K-means算法对句子进行聚类"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    return kmeans


def summarize_document(sentences, kmeans, X):
    """生成文档摘要，从每个聚类中选择最能代表该类的句子"""
    summary = []
    cluster_centers = kmeans.cluster_centers_
    for i in range(len(cluster_centers)):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_sentences = X[cluster_indices]
        centroid = cluster_centers[i]
        closest_index = np.argmax(cosine_similarity(cluster_sentences, centroid.reshape(1, -1)))
        summary.append(sentences[cluster_indices[closest_index]])
    return ' '.join(summary)


def generate_summary(input_file, output_file, n_clusters):
    """主函数，读取文档，进行预处理，聚类和生成摘要"""
    document = read_document(input_file)
    sentences = preprocess_text(document)
    X, vectorizer = compute_tfidf(sentences)
    kmeans = perform_clustering(X, n_clusters)
    summary = summarize_document(sentences, kmeans, X)

    with open(output_file, 'w') as file:
        file.write(summary)

    return summary


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("文档聚类与摘要生成")
        self.geometry("600x400")
        self.create_widgets()

    def create_widgets(self):
        # 文件选择按钮
        self.file_label = tk.Label(self, text="请选择输入文档:")
        self.file_label.pack(pady=10)

        self.file_button = tk.Button(self, text="选择文件", command=self.select_file)
        self.file_button.pack(pady=10)

        self.file_path_label = tk.Label(self, text="")
        self.file_path_label.pack(pady=10)

        # 聚类数输入
        self.cluster_label = tk.Label(self, text="请输入聚类数:")
        self.cluster_label.pack(pady=10)

        self.cluster_entry = tk.Entry(self)
        self.cluster_entry.pack(pady=10)

        # 生成摘要按钮
        self.generate_button = tk.Button(self, text="生成摘要", command=self.generate_summary_ui)
        self.generate_button.pack(pady=10)

        # 输出结果
        self.output_text = tk.Text(self, height=10)
        self.output_text.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_label.config(text=file_path)
            self.input_file = file_path
            messagebox.showinfo("文件选择", "文档读取成功！")

    def generate_summary_ui(self):
        try:
            n_clusters = int(self.cluster_entry.get())
            summary = generate_summary(self.input_file, "output_summary.txt", n_clusters)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, summary)
            messagebox.showinfo("生成摘要", "文档摘要生成成功！")
        except Exception as e:
            messagebox.showerror("错误", str(e))

if __name__ == "__main__":
    app = Application()
    app.mainloop()
# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603884 
- **Họ và tên:**  Nguyễn Minh Thành
- **SEED:**  5978 

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x] |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                                precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.809      0.894        0.947   
2. BoW + MLP (deep learning)         0.972   0.945      0.959        1.000   
3a. Encoder (MiniLM) + kNN           0.937   0.945      0.941        0.974   
3b. Encoder (MiniLM) + LogReg        0.973   0.973      0.973        1.000   
3c. Encoder (MiniLM) + MLP head      0.972   0.955      0.963        1.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            6.134      15965.042       0.054   
2. BoW + MLP (deep learning)        10.859      33336.949       0.307   
3a. Encoder (MiniLM) + kNN          81.921         14.101      54.176   
3b. Encoder (MiniLM) + LogReg       82.372         32.605      27.863   
3c. Encoder (MiniLM) + MLP head     82.918         32.821      28.646   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.107           0.0  
2. BoW + MLP (deep learning)          0.580           0.0  
3a. Encoder (MiniLM) + kNN           67.381           0.0  
3b. Encoder (MiniLM) + LogReg        37.985           0.0  
3c. Encoder (MiniLM) + MLP head      38.620           0.0  

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

1. Dự đoán hiệu năng của mạng MLP phi tuyến tính (dl_curve):
	• Bạn đoán: Mạng MLP phi tuyến tính sẽ có F1-score nhỉnh hơn một chút so với mô hình tuyến tính Logistic Regression ở phần 1.
	• Thực tế: F1-score trên tập test tăng vọt từ 0.894 (LogReg) lên 0.959 (MLP).
	• Vì sao lệch: Sự chênh lệch lớn hơn kỳ vọng ban đầu này xuất hiện do hàm kích hoạt phi tuyến tính ReLU kết hợp cấu trúc tầng ẩn (hidden = 64) đã giúp mạng Neural kết hợp và bóc tách các mối quan hệ đa từ (tổ hợp từ khóa xuất hiện đồng thời) rất mạnh mẽ, điều mà các trọng số độc lập tuyến tính của túi từ BoW đơn giản ở Phần 1 không thể tự tích hợp được.
2. Lợi ích của tầng ẩn khi tương tác trên không gian nhúng dày đặc (dl_relu / Phần 3):
	• Bạn đoán: Nếu xóa bỏ tính phi tuyến ReLU, mạng MLP 2 lớp sẽ suy biến toán học về dạng tuyến tính, làm F1-score sụt giảm hoặc không đổi.
	• Thực tế: Thử nghiệm không có ReLU cho kết quả F1 đạt 0.958. Đặc biệt trong phần nhúng, mô hình phân loại tuyến tính đơn giản 3b (LogReg) đạt F1 tới 0.973, cao hơn cả mô hình tầng ẩn phi tuyến 3c (MLP) đạt 0.963.
	• Vì sao lệch: Dự đoán ban đầu chưa tính đến bản chất của dữ liệu đầu vào. Đối với không gian nhúng dày đặc (Dense Embedding) như MiniLM, bản thân các đặc trưng vector trích xuất đã mang sẵn đặc tính hình học phi tuyến tính bậc cao và ngữ nghĩa phức tạp. Do đó, việc áp thêm một tầng ẩn phi tuyến dày đặc chỉ gây bão hòa hoặc quá khớp (overfitting), khiến một đường ranh giới phân tách tuyến tính phẳng (Linear/LogReg) là đã đủ tối ưu.
3. Tốc độ xử lý của mô hình Deep Learning (tin/s):
	• Bạn đoán: MLP có cấu trúc phức tạp và nhiều tham số hơn nên tốc độ tính toán (Inference) sẽ chậm hơn Logistic Regression truyền thống.
	• Thực tế: Khi chạy dự đoán cả lô (Batch), mô hình 2. BoW + MLP xử lý đạt hiệu suất kinh ngạc: 33,337 tin/s, nhanh gấp đôi so với 1. BoW + LogReg chỉ đạt 15,965 tin/s.
	• Vì sao lệch: PyTorch và NumPy tối ưu hóa các phép toán ma trận song song trên cấu trúc phần cứng cực tốt. Khi đẩy toàn bộ lô dữ liệu vào tầng ẩn, việc tính toán phân tán (Vectorized operations) giúp tận dụng tối đa luồng xử lý đồng thời, triệt tiêu độ trễ thô của các vòng lặp đơn lẻ.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Mô hình 1. BoW + LogReg có xu hướng bị sụt giảm độ chính xác nghiêm trọng nhất trên tập dữ liệu kiểm thử thực tế (Recall đạt thấp nhất: 0.809). Qua quá trình phân tích nhật ký lỗi (wrong cases), mô hình này liên tục bị đánh lừa bởi hai cơ chế bẫy ngôn ngữ chính:
• Bẫy 1: Từ viết tắt, tiếng lóng và biến thể ký tự (Né từ khóa - Obfuscation).
	• Tin nhắn lỗi: [thật=spam | đoán=ham ] Back 2 work 2morro half term over! Can U C me 2nite 4 some sexy passion B4 I have 2 go back? Chat NOW 09099726
	• Cơ chế sai: Phương pháp Bag-of-Words bẻ tách từ khóa rời rạc. Các từ như "2morro", "2nite", "B4" bị coi là những từ độc lập, nằm ngoài bộ từ điển huấn luyện phổ biến (vocab). Vì mô hình không thể kết nối ngữ nghĩa chuỗi, nó bỏ sót tín hiệu rác nguy hiểm ẩn sau dải số điện thoại tổng đài tính phí ở cuối câu.
• Bẫy 2: Tin rác sử dụng văn phong kể chuyện tự nhiên (Natural Spam / Storytelling).
	• Tin nhắn lỗi: [thật=spam | đoán=ham ] SMS. ac sun0819 posts HELLO:\You seem cool
	• Cơ chế sai: Tin nhắn mở đầu bằng các câu chào hỏi xã giao thông thường, có tần suất xuất hiện tương đồng với tập tin nhắn thường (ham). Do thiếu vắng các từ khóa kích hoạt rác trực diện như "free", "win", "claim", mô hình BoW hoàn toàn mất phương hướng và gán nhãn sai.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

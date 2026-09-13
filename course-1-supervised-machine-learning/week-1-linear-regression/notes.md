# Ghi Chú Tuần 1: Introduction to Machine Learning

---

## 1. Welcome to Machine Learning! (Chào mừng đến với ML)

### 💡 Định nghĩa cốt lõi (Core Definition)
> **"Machine Learning is the science of getting computers to learn without being explicitly programmed."**  
> *(Học máy là ngành khoa học giúp máy tính có khả năng tự học hỏi mà không cần phải được lập trình tường minh từng dòng lệnh).*

### 🌐 Ứng dụng thực tế quanh ta (Real-world Applications)
1. **Ứng dụng tiêu dùng hằng ngày (Consumer Apps):**
   - **Tìm kiếm web (Web Search):** Google, Bing, Baidu xếp hạng kết quả tìm kiếm (ranking web pages).
   - **Nhận diện khuôn mặt & Gắn thẻ:** Instagram, Snapchat tự động nhận biết bạn bè trong ảnh.
   - **Gợi ý nội dung (Recommender Systems):** Netflix / Streaming gợi ý phim tương tự sở thích.
   - **Nhận dạng giọng nói (Speech-to-Text / Assistants):** Siri, Google Assistant, chuyển giọng nói thành văn bản.
   - **Bộ lọc Spam (Spam Filtering):** Tự động phát hiện và chặn email lừa đảo / rác.

2. **Ứng dụng trong công nghiệp (Industrial & High-impact AI):**
   - **Biến đổi khí hậu & Năng lượng:** Tối ưu hóa hiệu suất phát điện của tua-bin gió.
   - **Y tế & Chăm sóc sức khỏe:** Hỗ trợ bác sĩ chẩn đoán bệnh chính xác từ hình ảnh/dữ liệu bệnh án.
   - **Sản xuất công nghiệp (Landing AI):** Sử dụng Computer Vision để kiểm tra lỗi sản phẩm tự động trên dây chuyền lắp ráp.

---

## 2. Applications of Machine Learning (Tại sao ML lại bùng nổ & Ứng dụng thực tế)

### ❓ Vì sao chúng ta cần Machine Learning thay vì Lập trình truyền thống?
- **Lập trình truyền thống (Traditional Software):** Con người tự tay viết các quy tắc (rules), ví dụ: thuật toán tìm đường đi ngắn nhất A* / Dijkstra trong GPS.
- **Ranh giới giới hạn:** Với những tác vụ phức tạp như: nhận diện giọng nói, đọc ảnh chụp X-quang, xe tự lái, xếp hạng tìm kiếm... con người **không thể nào ngồi viết hàng triệu dòng `if - else`** để bao quát mọi trường hợp.
- **Giải pháp duy nhất:** Cho máy tính **tự học quy luật từ dữ liệu (data)**.

### 🧠 AGI (Artificial General Intelligence - AI Tổng quát)
- **Định nghĩa:** Cỗ máy có trí thông minh và khả năng nhận thức ngang tầm hoặc vượt con người ở hầu hết mọi lĩnh vực.
- **Quan điểm của Andrew Ng:** AGI đang bị thổi phồng (overhyped) và còn rất xa (có thể 50, 500 năm nữa hoặc hơn). Tuy nhiên, con đường khả thi nhất để tiến gần tới AGI chính là **các thuật toán học (Learning Algorithms)**, đặc biệt là những thuật toán mô phỏng cách não bộ con người hoạt động (Neural Networks).

### 💰 Giá trị kinh tế & Cơ hội nghề nghiệp
- Báo cáo McKinsey dự báo AI/ML sẽ tạo ra thêm **13 nghìn tỷ USD** giá trị hàng năm vào năm 2030.
- Giá trị khổng lồ chưa được khai phá nằm ở **các ngành công nghiệp phi-phần mềm (non-software sectors)**: Nông nghiệp, Y tế, Bán lẻ, Giao thông, Chế tạo ô tô, Vật liệu...
- Nhu cầu kỹ sư hiểu sâu lý thuyết và biết tự tay code (thực hành) thuật toán đang cực kỳ thiếu hụt.

---

## 3. What is Machine Learning? (Định nghĩa & Tư duy áp dụng)

### ♟️ Định nghĩa Arthur Samuel (1959)
> *"Field of study that gives computers the ability to learn without being explicitly programmed."*

- **Câu chuyện lịch sử:** Chương trình cờ đam (Checkers) của Arthur Samuel.
  - Bản thân Samuel chơi cờ đam không giỏi.
  - Ông lập trình cho máy tính **tự đấu hàng chục ngàn ván cờ với chính nó**.
  - Nhờ quan sát những thế cờ nào thường dẫn đến chiến thắng, thế cờ nào dẫn đến thất bại, máy tính tích lũy kinh nghiệm qua từng ván và dần dần trở thành một đối thủ cờ đam siêu hạng, **giỏi hơn nhiều so với chính tác giả**.
- **Nguyên lý quan trọng:** Càng cung cấp nhiều cơ hội (dữ liệu & lượt chơi) để thuật toán học tập, mô hình sẽ hoạt động càng tốt.

### 🗂️ Hai nhánh thuật toán học máy chính:
1. **Supervised Learning (Học có giám sát):**
   - Phổ biến nhất trong các ứng dụng thực tế hiện nay, đạt được nhiều bước tiến đột phá nhất (trọng tâm của Course 1 & 2).
2. **Unsupervised Learning (Học không giám sát):**
   - Khám phá các mẫu hoặc cấu trúc tiềm ẩn từ dữ liệu chưa được gắn nhãn (trọng tâm Course 3, cùng Recommender Systems và Reinforcement Learning).

### 🛠️ Lời khuyên thực chiến (Practical Advice)
- Học thuật toán giống như có trong tay một bộ đồ nghề (búa, máy khoan xịn). Nhưng có đồ nghề không có nghĩa là bạn biết cách xây một ngôi nhà 3 tầng kiên cố.
- Khóa học này chú trọng cả 2 phần: **Hiểu công cụ (Algorithms)** và **Kỹ năng áp dụng thực tế (Best Practices)** để tránh mất hàng tháng trời đi chệch hướng trong các dự án AI thực tế.

---

## 4. Supervised Learning: Part 1 — Regression (Học có giám sát & Bài toán Hồi quy)

### 🎯 Bản chất của Supervised Learning
> Học ánh xạ từ đầu vào đến đầu ra: **$x \rightarrow y$ (Input to Output mapping)**.

- **Đặc điểm then chốt:** Cung cấp cho thuật toán tập dữ liệu huấn luyện đã có sẵn **"câu trả lời đúng" (right answers / labels $y$)** tương ứng với từng đầu vào $x$.
- Sau khi được huấn luyện qua nhiều cặp $(x, y)$, mô hình có khả năng nhận một giá trị $x$ hoàn toàn mới và đưa ra dự đoán chính xác giá trị $y$.

### 📊 Bảng ví dụ ánh xạ $x \rightarrow y$ trong thực tế:
| Đầu vào $x$ (Input) | Nhãn đầu ra $y$ (Output) | Ứng dụng (Application) |
| :--- | :--- | :--- |
| Email | Spam (0 hoặc 1) | Bộ lọc thư rác (Spam Filter) |
| Đoạn âm thanh (Audio clip) | Bản ghi văn bản (Text transcript) | Nhận dạng giọng nói (Speech Recognition) |
| Văn bản tiếng Anh | Văn bản tiếng Việt / Tây Ban Nha | Dịch máy (Machine Translation) |
| Thông tin user + Thông tin quảng cáo | Click / Không Click (0 hoặc 1) | Quảng cáo trực tuyến (Online Ad Clicks) |
| Ảnh camera + Cảm biến Radar | Vị trí các xe xung quanh | Xe tự hành (Self-driving car) |
| Ảnh sản phẩm vừa xuất xưởng | Có trầy/móp/lỗi hay không | Kiểm định chất lượng thị giác (Visual Inspection) |
| Diện tích nhà (Square feet / $m^2$) | Giá bán căn nhà ($) | Định giá bất động sản |

### 📈 Bài toán Hồi quy (Regression) là gì?
- **Định nghĩa:** Hồi quy là dạng bài toán học có giám sát trong đó mô hình cần **dự đoán một giá trị số liên tục từ vô số các giá trị khả dĩ (predict a number from infinitely many possible numbers)**.
- **Ví dụ điển hình:** Dự đoán giá nhà dựa trên diện tích. Giá nhà có thể là 150k, 150.5k, 200k hoặc bất kỳ con số nào ở giữa.
- **Cách tiếp cận:** Thuật toán sẽ tìm cách khớp một hàm số phù hợp đi qua các điểm dữ liệu:
  - Khớp đường thẳng (Straight line fit) $\rightarrow$ *Linear Regression*.
  - Hoặc khớp đường cong (Curve / Polynomial fit) $\rightarrow$ *Non-linear / Polynomial Regression*.




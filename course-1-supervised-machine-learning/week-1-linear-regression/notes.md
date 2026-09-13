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

---

## 5. Supervised Learning: Part 2 — Classification (Học có giám sát & Bài toán Phân loại)

### 🏷️ Phân loại (Classification) là gì?
- **Định nghĩa:** Phân loại là bài toán dự đoán **các danh mục rời rạc (categories / classes)** từ một tập hữu hạn các giá trị có thể có (thay vì dự đoán một dải số liên tục như Hồi quy).
- Các thuật ngữ tương đương trong tài liệu và bài giảng: **Class** = **Category** = **Label** (Lớp / Danh mục / Nhãn).

### 🩺 Ví dụ kinh điển: Chẩn đoán ung thư vú (Breast Cancer Detection)
- **Mục tiêu:** Dự đoán một khối u (tumor) là:
  - `0`: Lành tính (**Benign** — không phải ung thư, an toàn), thường ký hiệu hình tròn `O`.
  - `1`: Ác tính (**Malignant** — ung thư, nguy hiểm), thường ký hiệu dấu `X`.
- Đây là **Binary Classification (Phân loại nhị phân)**: chỉ có 2 khả năng đầu ra ($0$ hoặc $1$).

### 🔢 Phân loại nhiều lớp (Multi-class Classification)
- Không chỉ giới hạn ở 2 lớp, bài toán có thể có 3, 4 hoặc nhiều lớp hơn:
  - Ví dụ ung thư: Type 0, Type 1, Type 2.
  - Nhận diện động vật trong ảnh: Chó, Mèo, Gấu trúc, Ngựa...
  - **Điểm khác biệt quan trọng với Regression:** Khi đầu ra được mã hóa bằng số $(0, 1, 2)$, mô hình phân loại **chỉ dự đoán đúng các giá trị nguyên rời rạc này**, KHÔNG BAO GIỜ mang ý nghĩa các số nằm giữa như $0.5$ hay $1.7$.

### 📐 Nhiều biến đầu vào (Multiple Features) & Đường ranh giới (Decision Boundary)
- **1 đặc trưng ($x_1$):** Chỉ dựa vào kích thước khối u (Tumor size).
- **2 đặc trưng ($x_1, x_2$):** Dựa vào Tuổi bệnh nhân (Age) và Kích thước khối u (Tumor size). Dữ liệu được vẽ trên hệ tọa độ 2D.
- **Nhiều đặc trưng thực tế ($x_1, x_2, ..., x_n$):** Độ dày khối u, độ đồng đều hình dạng tế bào, viền tế bào...
- **Nhiệm vụ của thuật toán:** Tìm ra một **Đường ranh giới quyết định (Decision Boundary)** để phân tách tối ưu giữa các lớp (ví dụ tách vùng Lành tính ra khỏi vùng Ác tính). Khi một bệnh nhân mới đến, dựa vào vị trí của họ so với đường ranh giới, bác sĩ có thể đưa ra kết luận chẩn đoán.

---

### ⚖️ So sánh tóm tắt: Regression vs. Classification
| Tiêu chí | Regression (Hồi quy) | Classification (Phân loại) |
| :--- | :--- | :--- |
| **Đầu ra ($y$)** | Số liên tục (Continuous number) | Lớp rời rạc (Discrete category / class) |
| **Không gian giá trị** | Vô số giá trị khả dĩ (Infinitely many values) | Tập hợp nhỏ, hữu hạn các lớp ($2, 3, ...$) |
| **Ví dụ bài toán** | Dự đoán giá nhà, nhiệt độ, doanh số bán lẻ | Spam / Not Spam, Ung thư ác tính / Lành tính, Chó / Mèo |
| **Mô hình tìm kiếm** | Khớp đường xu hướng (Fit curve / line) | Tìm đường ranh giới phân tách (Decision boundary) |

---

## 6. Unsupervised Learning: Part 1 — Clustering (Học không giám sát & Thuật toán Phân cụm)

### 🧩 Bản chất của Unsupervised Learning (Học không giám sát)
> Khác với Supervised Learning (luôn có cặp $(x, y)$ kèm nhãn đúng $y$), Unsupervised Learning được cung cấp dữ liệu **KHÔNG CÓ BẤT KỲ NHÃN ĐÁP ÁN ĐÚNG NÀO ($y$)**.

- **Đầu vào:** Chỉ có các đặc trưng $x$ (ví dụ: tuổi và kích thước khối u, nhưng không cho biết khối u đó là lành hay ác tính).
- **Mục tiêu:** Thuật toán tự khám phá cấu trúc tiềm ẩn (structure), phát hiện các quy luật (patterns) hoặc những thông tin thú vị ẩn giấu trong dữ liệu mà không có con người giám sát hay chỉ dẫn trước.

### 🌐 Thuật toán Phân cụm (Clustering Algorithm)
- **Định nghĩa:** Là dạng thuật toán Unsupervised Learning phổ biến nhất. Thuật toán tự động nhóm các điểm dữ liệu tương đồng thành các **cụm (clusters)** tách biệt.
- **3 Ứng dụng thực tế nổi bật:**

1. **Google News (Gom nhóm tin tức tự động):**
   - Hàng trăm ngàn bài báo xuất hiện mỗi ngày trên Internet.
   - Thuật toán phân cụm tự động quét các từ khóa xuất hiện cùng nhau (ví dụ: *"panda"*, *"twin"*, *"zoo"*) trên nhiều trang báo để tự động gom lại thành một cụm chủ đề duy nhất (ví dụ: *Gấu trúc sinh đôi ở vườn thú Nhật Bản*). Không cần bất kỳ con người nào gán nhãn thủ công mỗi ngày.

2. **Dữ liệu di truyền học & ADN (DNA Microarray Data):**
   - Mỗi cột đại diện cho ADN của một cá nhân, mỗi hàng là mức độ biểu hiện (expression) của một gen cụ thể (màu mắt, chiều cao, thậm chí gen quy định việc ghét ăn súp lơ xanh / broccoli...).
   - Chạy thuật toán phân cụm giúp tự động phân loại các cá nhân thành các nhóm sinh học tương đồng (Type 1, Type 2, Type 3...) mà không cần bác sĩ hay nhà khoa học phải định nghĩa trước đặc điểm từng nhóm.

3. **Phân khúc thị trường & Người dùng (Market Segmentation):**
   - Phân tích hàng triệu khách hàng trong cơ sở dữ liệu để tìm ra các nhóm hành vi tương đồng.
   - *Ví dụ thực tế từ DeepLearning.AI:* Phân cụm học viên theo mục đích học:
     - Nhóm 1: Học để nâng cao kiến thức & thỏa mãn đam mê.
     - Nhóm 2: Học để chuyển nghề hoặc thăng tiến công việc.
     - Nhóm 3: Học để cập nhật tác động của AI đối với ngành nghề hiện tại của mình.






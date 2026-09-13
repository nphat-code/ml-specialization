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

---

## 7. Unsupervised Learning: Part 2 — Anomaly Detection & Dimensionality Reduction

### 🎯 3 Dạng Thuật toán Học không giám sát chính (Khóa 3 sẽ học sâu)

1. **Clustering (Phân cụm):**
   - Tự động gom các điểm dữ liệu tương đồng thành các nhóm tách biệt.
2. **Anomaly Detection (Phát hiện điểm bất thường):**
   - Phát hiện các sự kiện khác biệt bất thường so với dữ liệu thông thường.
   - **Ứng dụng thực tế lớn nhất:** Phát hiện gian lận tài chính (Financial Fraud Detection) trong hệ thống ngân hàng khi có một giao dịch bất thường xảy ra; phát hiện hỏng hóc thiết bị trong dây chuyền sản xuất công nghiệp.
3. **Dimensionality Reduction (Giảm chiều dữ liệu - ví dụ PCA):**
   - Cho phép nén một tập dữ liệu lớn với hàng trăm/hàng ngàn đặc trưng (features) về một tập dữ liệu nhỏ gọn hơn rất nhiều mà **bảo toàn tối đa lượng thông tin**.
   - Giúp giảm dung lượng lưu trữ, tăng tốc độ huấn luyện mô hình và trực quan hóa dữ liệu trên đồ thị 2D/3D.

---

### 📝 Câu hỏi kiểm tra trực giác (Quiz Drill - Rất hay thi!)

Hãy xác định các bài toán sau thuộc loại học máy nào:

| Bài toán thực tế | Loại học máy | Giải thích bản chất |
| :--- | :--- | :--- |
| **Bộ lọc Spam (Spam Filtering)** | **Supervised Learning** *(Classification)* | Dữ liệu huấn luyện có sẵn nhãn email là Spam ($y=1$) hay Non-Spam ($y=0$). |
| **Gom nhóm tin tức Google News** | **Unsupervised Learning** *(Clustering)* | Hàng triệu bài báo không có sẵn nhãn chủ đề, thuật toán tự gom các bài cùng từ khóa vào 1 nhóm. |
| **Phân khúc thị trường (Market Segmentation)** | **Unsupervised Learning** *(Clustering)* | Dữ liệu hồ sơ khách hàng không có nhãn sẵn, thuật toán tự phân chia thành các nhóm khách hàng hành vi tương đồng. |
| **Chẩn đoán bệnh tiểu đường (Diagnosing Diabetes)** | **Supervised Learning** *(Classification)* | Giống bài toán ung thư vú: có hồ sơ bệnh án đầu vào $x$ và nhãn kết quả xét nghiệm $y$ (Mắc tiểu đường / Không mắc). |

---

## 8. Jupyter Notebooks (Môi trường thực hành chuẩn của ML & Data Science)

### 💻 Jupyter Notebook là gì?
- Công cụ tương tác tiêu chuẩn số 1 thế giới được các kỹ sư AI và nhà khoa học dữ liệu sử dụng để viết code, chạy thử nghiệm và vẽ đồ thị trực quan.
- Chạy được trực tiếp trên trình duyệt (Coursera Labs) hoặc tích hợp hoàn hảo trong VS Code.

### 🧱 2 Khối cơ bản (Cells) cần nắm:
1. **Markdown Cell (Ô văn bản):**
   - Dùng để ghi chú, giải thích lý thuyết, chèn công thức toán LaTeX.
   - Nhấn **`Shift + Enter`** để render văn bản đẹp mắt.
2. **Code Cell (Ô mã nguồn):**
   - Chứa các đoạn code Python có thể thực thi.
   - Nhấn **`Shift + Enter`** để chạy đoạn code và in kết quả/vẽ biểu đồ ngay bên dưới.

### 🧪 2 Loại bài Lab trong khóa học:
- **Optional Labs (Lab tự chọn / làm quen):**
  - Code đã được viết sẵn hoàn chỉnh từ đầu đến cuối.
  - Mục tiêu: Quan sát cách viết code ML thực tế, bấm `Shift + Enter` chạy qua từng cell để cảm nhận luồng dữ liệu, tự do nghịch và sửa đổi tham số để xem kết quả thay đổi.
  - Không chấm điểm (không áp lực).
- **Practice Labs / Graded Assignments (Bắt đầu từ tuần 2):**
  - Học viên sẽ tự tay viết thêm code logic thuật toán vào các ô trống quy định để nộp bài chấm điểm.

---

## 9. Linear Regression: Part 1 — Model & Notations (Mô hình Hồi quy tuyến tính & Ký hiệu chuẩn)

### 📈 Linear Regression là gì?
- Là việc **khớp một đường thẳng (fitting a straight line)** vào tập dữ liệu quan sát được.
- Là thuật toán học máy phổ biến và nền tảng nhất trên thế giới. Mọi nguyên lý toán học và tư duy ở đây sẽ được tái sử dụng trong các mô hình phức tạp hơn (Neural Networks / Deep Learning).

### 📐 Ký hiệu toán học quy chuẩn trong Machine Learning (Terminology & Notation)
| Ký hiệu | Tên gọi tiếng Anh | Tên gọi tiếng Việt | Ví dụ trong bài toán giá nhà Portland |
| :---: | :--- | :--- | :--- |
| **$x$** | Input variable / Feature | Biến đầu vào / Đặc trưng | Diện tích căn nhà ($2104\text{ sq ft}$) |
| **$y$** | Output variable / Target | Biến đầu ra / Mục tiêu | Giá bán căn nhà ($\$400k$) |
| **$m$** | Number of training examples | Tổng số mẫu huấn luyện | $m = 47$ căn nhà trong bảng dữ liệu |
| **$(x, y)$** | Single training example | Một mẫu huấn luyện đơn lẻ | Cặp giá trị $(2104, 400)$ |
| **$(x^{(i)}, y^{(i)})$** | $i^{\text{th}}$ training example | Mẫu huấn luyện thứ $i$ (dòng thứ $i$) | $(x^{(1)}, y^{(1)}) = (2104, 400)$ |

### ⚠️ Lưu ý sống còn về ký hiệu chỉ số trên $(i)$:
- Ký hiệu $x^{(i)}$ có dấu ngoặc tròn là **chỉ số dòng thứ $i$ trong bảng dữ liệu (index into training set)**.
- **TUYỆT ĐỐI KHÔNG PHẢI LŨY THỪA / SỐ MŨ (NOT exponentiation)**.
  - $x^{(2)}$ là đặc trưng của ngôi nhà thứ 2 trong bảng, KHÔNG PHẢI $x^2$ (bình phương).
  - $y^{(i)}$ là giá bán thực tế của ngôi nhà thứ $i$.

---

## 10. Linear Regression: Part 2 — Supervised Learning Process & Model Representation

### 🔄 Luồng hoạt động của Supervised Learning (The Supervised Learning Pipeline)
Quy trình cốt lõi gồm 4 bước:

```text
  [Training Set] 
  (Features x, Targets y)
         │
         ▼
[Learning Algorithm] ─────► Sinh ra Hàm mô hình f (Model / Hypothesis)
                                  │
      Đầu vào mới x ──────────────┼──────────► Dự đoán y-hat (ŷ = f(x))
```

1. **Training Set:** Cung cấp các đặc trưng $x$ và nhãn mục tiêu thực tế $y$ (câu trả lời đúng).
2. **Learning Algorithm:** Học các quy luật từ tập dữ liệu huấn luyện.
3. **Model $f$ (Function / Hypothesis):** Mô hình được sinh ra từ thuật toán.
4. **Dự đoán $\hat{y}$ (y-hat):** Khi có một ngôi nhà mới với diện tích $x$, nạp vào hàm $f$ để tính ra giá trị dự đoán $\hat{y} = f(x)$.

### ⚖️ Phân biệt $y$ vs. $\hat{y}$ (y-hat):
- **$y$:** Giá trị thực tế (True value / Ground truth target) trong tập dữ liệu.
- **$\hat{y}$ ($y$ có dấu mũ trên đầu - "y-hat"):** Giá trị ước lượng / dự đoán của mô hình (Estimate / Prediction). $\hat{y}$ có thể bằng hoặc có sai số so với $y$.

### 📐 Biểu diễn toán học của Mô hình (Model Representation):
Hàm hồi quy tuyến tính đơn biến (Univariate Linear Regression):
$$f_{w,b}(x) = wx + b$$
*(hoặc viết gọn là $f(x) = wx + b$)*

- **$x$:** Biến đầu vào (Feature).
- **$w, b$:** Các tham số của mô hình (**Parameters** hay **Weights / Bias**).
  - Giá trị của $w$ (hệ số góc/độ dốc) và $b$ (điểm cắt trục tung) sẽ quyết định hình dạng và vị trí của đường thẳng.
- **Univariate Linear Regression:**
  - *Uni* = Một (Latinh).
  - *Variate* = Biến.
  - $\rightarrow$ Hồi quy tuyến tính **đơn biến** (chỉ có duy nhất 1 biến đầu vào $x$).

---

## 11. Cost Function Formula (Công thức Hàm Chi Phí / Bình phương sai số)

### 🎯 Vai trò của Cost Function
- Khi huấn luyện mô hình $f_{w,b}(x) = wx + b$, $w$ và $b$ là các **tham số (parameters / weights & bias)** mà ta có thể điều chỉnh.
- Thay đổi $w$ và $b$ sẽ tạo ra các đường thẳng khác nhau:
  - $w$ quyết định **độ dốc (slope)** của đường thẳng.
  - $b$ quyết định **điểm cắt trục tung (y-intercept)** khi $x = 0$.
- **Câu hỏi cốt lõi:** Làm thế nào để máy tính tự đo xem đường thẳng nào "khớp tốt nhất" với dữ liệu? $\rightarrow$ Ta cần xây dựng **Hàm chi phí $J(w, b)$**.

### 📐 Từng bước xây dựng công thức $J(w, b)$:
1. **Sai số tại mẫu thứ $i$ (Error):**
   $$\text{error}^{(i)} = \hat{y}^{(i)} - y^{(i)} = f_{w,b}(x^{(i)}) - y^{(i)}$$
2. **Bình phương sai số (Squared Error):**
   $$(\hat{y}^{(i)} - y^{(i)})^2$$
   - Giúp sai số luôn $\ge 0$ (dù đoán cao hơn hay thấp hơn giá trị thật đều bị tính là lỗi).
   - Phạt nặng hơn các dự đoán sai lệch lớn.
3. **Tổng bình phương sai số của toàn bộ $m$ mẫu:**
   $$\sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2$$
4. **Chia cho $m$ để lấy trung bình:**
   - Giúp hàm chi phí không bị tự động phình to khi số lượng mẫu $m$ tăng lên.
5. **Chia thêm cho $2$:**
   - Quy ước chuẩn giúp phép tính đạo hàm sau này triệt tiêu gọn với số mũ 2:
   $$J(w, b) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

### 🏆 Mục tiêu của giải thuật (Optimization Goal):
$$\min_{w, b} J(w, b)$$
Tìm cặp giá trị $(w, b)$ sao cho hàm chi phí $J(w, b)$ đạt giá trị **nhỏ nhất có thể**.

---

## 12. Cost Function Intuition (Trực giác về Hàm Chi Phí qua mô hình đơn giản b = 0)

### 🎯 Mô hình đơn giản hóa (Simplified Model):
- Để dễ trực quan hóa trước khi xét đồng thời cả $w$ và $b$, ta tạm gán $b = 0$:
  $$f_w(x) = w \cdot x$$
- Lúc này đường thẳng luôn đi qua gốc tọa độ $(0, 0)$.
- Hàm chi phí chỉ phụ thuộc vào một biến duy nhất là $w$:
  $$J(w) = \frac{1}{2m} \sum_{i=1}^m (w \cdot x^{(i)} - y^{(i)})^2$$

### 📊 Mối liên hệ giữa 2 đồ thị song song (Side-by-side Visuals):
Giả sử tập dữ liệu có $m = 3$ điểm: $(1, 1), (2, 2), (3, 3)$.

```text
    Đồ thị Mô hình f(x)             Đồ thị Hàm chi phí J(w)
          (x vs y)                           (w vs J(w))

   y ^                                  J(w) ^
   3 |       *(3,3)                        6 |      * (w=-0.5, J≈5.25)
   2 |    *(2,2)                           4 |
   1 | *(1,1)                              2 |   * (w=0, J≈2.33)
     +─────────────> x                     0 |───*──────*─────────> w
     0   1   2   3                           -0.5 0    0.5  1 (w=1, J=0: Đáy Parabol)
```

### 🔢 Tính toán chi tiết qua từng giá trị của $w$:
1. **Khi chọn $w = 1$:**
   - $f(x) = 1 \cdot x = x \implies$ Đi qua chính xác cả 3 điểm $(1,1), (2,2), (3,3)$.
   - Sai số tại mọi điểm đều bằng $0$.
   - $J(1) = \frac{1}{2 \times 3} (0^2 + 0^2 + 0^2) = 0$.
   - Điểm trên đồ thị $J$: **$(w = 1, J = 0)$** $\rightarrow$ **Điểm cực tiểu hoàn hảo!**

2. **Khi chọn $w = 0.5$:**
   - $f(x) = 0.5x$ (đường thẳng thoải hơn).
   - Điểm 1 ($x=1$): $f(1) = 0.5 \implies$ sai số: $(0.5 - 1)^2 = 0.25$.
   - Điểm 2 ($x=2$): $f(2) = 1.0 \implies$ sai số: $(1.0 - 2)^2 = 1.00$.
   - Điểm 3 ($x=3$): $f(3) = 1.5 \implies$ sai số: $(1.5 - 3)^2 = 2.25$.
   - Tổng sai số: $0.25 + 1.00 + 2.25 = 3.5$.
   - $J(0.5) = \frac{3.5}{2 \times 3} = \frac{3.5}{6} \approx 0.58$.
   - Điểm trên đồ thị $J$: **$(w = 0.5, J \approx 0.58)$**.

3. **Khi chọn $w = 0$:**
   - $f(x) = 0$ (đường nằm ngang trùng trục hoành).
   - Tổng bình phương sai số: $1^2 + 2^2 + 3^2 = 14$.
   - $J(0) = \frac{14}{6} \approx 2.33$.
   - Điểm trên đồ thị $J$: **$(w = 0, J \approx 2.33)$**.

4. **Khi chọn $w = -0.5$:**
   - $f(x) = -0.5x$ (đường dốc ngược xuống).
   - $J(-0.5) \approx 5.25$ (sai số rất lớn).

### 💡 Kết luận trực giác cốt lõi:
- Đồ thị của $J(w)$ có hình dạng **Parabol (hình cái chảo / chữ U)**.
- Mỗi giá trị tham số $w$ quyết định một đường thẳng $f(x)$ bên trái, tương ứng với **một điểm duy nhất** trên đồ thị $J(w)$ bên phải.
- Đường thẳng càng khớp sát dữ liệu $\rightarrow$ Giá trị $J(w)$ càng nằm sâu xuống đáy chữ U.
- Tại đáy thung lũng ($w = 1$), hàm chi phí đạt cực tiểu $J = 0$.












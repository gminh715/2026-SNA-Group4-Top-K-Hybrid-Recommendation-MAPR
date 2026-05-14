# Tóm tắt nghiên cứu: Hệ gợi ý phim kết hợp LightFM, biểu diễn văn bản và đồ thị tri thức

## Tóm tắt

Nghiên cứu này xây dựng và đánh giá một hệ gợi ý phim trong bối cảnh phản hồi ẩn (implicit feedback). Rating gốc được chuyển thành tương tác dương khi thỏa điều kiện $r_{u,i} \geq 7$, sau đó các mô hình được so sánh trong cùng một giao thức chia dữ liệu Leave-One-Out.

Mục tiêu chính là đánh giá hiệu quả của mô hình **SeRel-LightFM**, một biến thể mở rộng của LightFM có khả năng kết hợp ba nguồn tín hiệu: metadata phim, embedding văn bản và embedding đồ thị tri thức (Knowledge Graph Embedding, KGE). Kết quả được báo cáo trên **full TEST** và chỉ sử dụng hai nhóm chỉ số: **Recall@$K$** và **NDCG@$K$**.

Các mô hình được so sánh gồm:

- **Popularity Baseline:** xếp hạng phim theo số lượng tương tác dương trong tập huấn luyện.
- **ItemKNN Baseline:** lọc cộng tác dựa trên độ tương đồng giữa các item.
- **LightFM Baseline:** LightFM sử dụng đặc trưng item và user cơ bản.
- **Ablation không Text Encoder:** loại bỏ embedding văn bản khỏi SeRel-LightFM.
- **Ablation không KGE:** loại bỏ embedding đồ thị tri thức khỏi SeRel-LightFM.
- **SeRel-LightFM đầy đủ:** kết hợp metadata, embedding văn bản và embedding đồ thị tri thức.

## 1. Bối cảnh và mục tiêu nghiên cứu

Bài toán gợi ý phim cần xếp hạng các item phù hợp nhất cho từng người dùng dựa trên lịch sử tương tác và thông tin bổ trợ. Trong nghiên cứu này, bài toán được tiếp cận theo hướng xếp hạng Top-$K$, tức mô hình cần đưa item phù hợp lên các vị trí cao trong danh sách đề xuất.

Các câu hỏi nghiên cứu chính:

- LightFM có cải thiện chất lượng gợi ý so với Popularity Baseline và ItemKNN hay không?
- Embedding văn bản có đóng góp như thế nào khi được đưa vào LightFM dưới dạng cụm đặc trưng?
- Embedding đồ thị tri thức có bổ sung tín hiệu hữu ích cho mô hình gợi ý hay không?
- SeRel-LightFM đầy đủ có đạt Recall và NDCG tốt hơn các baseline và các biến thể ablation hay không?

## 2. Dữ liệu và thiết lập thực nghiệm

Dữ liệu gồm ba bảng chính:

- `ratings.csv`: chứa tương tác người dùng-phim và rating gốc.
- `movies.csv`: chứa metadata phim như thể loại, chủ đề, quốc gia, năm phát hành, thời lượng, đạo diễn, diễn viên và nội dung mô tả.
- `user_profiles.csv`: chứa hồ sơ tĩnh và các đặc trưng hành vi của người dùng.

Các bước tiền xử lý chính:

- Loại bỏ rating của user không có hồ sơ.
- Chỉ giữ user có đủ số lượng rating tối thiểu để phục vụ chia dữ liệu.
- Loại bỏ rating trỏ tới movie không tồn tại trong metadata.
- Chuẩn hóa các cột văn bản để giảm lỗi encoding và thống nhất định dạng token.

Thiết lập chia dữ liệu sử dụng Leave-One-Out theo từng người dùng:

- **Train:** toàn bộ tương tác cũ hơn.
- **Validation:** tương tác ngay trước tương tác kiểm thử.
- **Test:** tương tác mới nhất của mỗi người dùng.

Sau khi áp dụng ngưỡng phản hồi ẩn $r_{u,i} \geq 7$, full TEST có 262 người dùng được dùng trong đánh giá.

## 3. Phương pháp

### 3.1 Popularity Baseline

Popularity Baseline tính điểm phổ biến của item $i$ bằng số lượng tương tác dương trong tập huấn luyện:

$$
\text{pop}(i) = \sum_u \mathbb{1}(r_{u,i} \geq \tau),
$$

trong đó $\tau = 7$. Khi sinh gợi ý, mô hình sắp xếp item theo $\text{pop}(i)$ giảm dần và loại bỏ các item đã xuất hiện trong lịch sử huấn luyện của người dùng.

Đây là mốc so sánh đơn giản nhất, giúp kiểm tra xem các mô hình học máy có vượt qua được tín hiệu phổ biến toàn cục hay không.

### 3.2 ItemKNN Baseline

ItemKNN xây dựng ma trận tương tác nhị phân:

$$
R_{u,i} = \mathbb{1}(r_{u,i} \geq \tau).
$$

Độ tương đồng giữa hai item được tính bằng cosine similarity trên vector tương tác người dùng. Với user $u$ và item ứng viên $j$, điểm gợi ý được tính bằng tổng độ tương đồng giữa $j$ và các item mà user đã tương tác dương trong train:

$$
\mathrm{score}(u,j) =
\sum_{i \in N(j,K) \cap I_u^+} \mathrm{sim}(i,j).
$$

Trong đó $N(j,K)$ là tập $K$ item lân cận của item $j$, còn $I_u^+$ là tập item dương trong lịch sử huấn luyện của user $u$.

### 3.3 LightFM Baseline

LightFM Baseline sử dụng hàm mất mát WARP để học mô hình xếp hạng. Mô hình kết hợp tương tác user-item với đặc trưng phụ:

- Đặc trưng item: `genre`, `topic`, `country`, `year`, `duration`.
- Đặc trưng user: các cờ hành vi, nhóm giờ ưa thích, năm tạo tài khoản và ngày trong tuần.

Baseline này chưa sử dụng embedding văn bản hoặc embedding đồ thị tri thức. Do đó, nó đóng vai trò mốc so sánh chính cho SeRel-LightFM.

### 3.4 SeRel-LightFM

SeRel-LightFM mở rộng LightFM bằng cách bổ sung các cụm đặc trưng được tạo từ embedding văn bản và embedding đồ thị tri thức.

| Nhóm đặc trưng | Nguồn tín hiệu | Cách đưa vào LightFM |
|---|---|---|
| Metadata item | Thể loại, chủ đề, quốc gia, năm phát hành, thời lượng | Feature rời rạc và biến liên tục đã chuẩn hóa |
| Hồ sơ user | Thông tin tĩnh và hành vi tổng quát | Feature rời rạc phía user |
| Embedding văn bản | Sentence Transformer trên mô tả đa trường của phim | KMeans text clusters |
| Embedding đồ thị tri thức | TransE trên đồ thị tri thức mở rộng | KMeans KGE clusters |

#### Biểu diễn văn bản

Các trường văn bản của phim được ghép thành đầu vào đa trường, sau đó mã hóa bằng Sentence Transformer. Embedding được chuẩn hóa $L_2$ và gom cụm bằng KMeans. Với mỗi item, các cụm gần nhất được gán trọng số bằng softmax:

$$
w_c =
\frac{\exp(s_c / T)}
{\sum_{c'} \exp(s_{c'} / T)},
$$

trong đó $s_c$ là độ tương đồng giữa embedding item và centroid cụm $c$, còn $T$ là temperature.

#### Biểu diễn đồ thị tri thức

Đồ thị tri thức được xây dựng từ metadata phim, hồ sơ người dùng và tương tác user-item trong tập huấn luyện. TransE học embedding sao cho mỗi bộ ba $(h,r,t)$ thỏa gần đúng:

$$
\mathbf{h} + \mathbf{r} \approx \mathbf{t}.
$$

Embedding của item và user sau huấn luyện được chuẩn hóa, gom cụm bằng KMeans và đưa vào LightFM như các feature rời rạc có trọng số.

## 4. Thiết kế ablation

Hai biến thể ablation được dùng để phân tích đóng góp của từng thành phần trong SeRel-LightFM.

### 4.1 Ablation không KGE

Biến thể này loại bỏ toàn bộ nhánh đồ thị tri thức, gồm xây dựng KG, huấn luyện TransE và tạo KGE clusters. Mô hình vẫn giữ metadata, user features và text clusters. Mục tiêu là đo tác động của KGE khi embedding văn bản vẫn được sử dụng.

### 4.2 Ablation không Text Encoder

Biến thể này loại bỏ Sentence Transformer và text clusters. Mô hình vẫn giữ metadata, user features và KGE clusters. Mục tiêu là đo tác động của embedding văn bản khi tín hiệu đồ thị tri thức vẫn được sử dụng.

## 5. Chỉ số đánh giá

Nghiên cứu chỉ sử dụng hai chỉ số trên full TEST:

### 5.1 Recall@$K$

Recall@$K$ đo tỷ lệ item đúng được tìm thấy trong top-$K$ gợi ý:

$$
\mathrm{Recall@K} =
\frac{|G_u \cap P_u^K|}{|G_u|},
$$

trong đó $G_u$ là tập item đúng của user $u$, còn $P_u^K$ là tập top-$K$ item được mô hình đề xuất.

### 5.2 NDCG@$K$

NDCG@$K$ đánh giá chất lượng thứ hạng của item đúng trong top-$K$. Chỉ số này ưu tiên các mô hình đưa item đúng lên vị trí cao hơn:

$$
\mathrm{NDCG@K} =
\frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}.
$$

Với bài toán gợi ý Top-$K$, NDCG thường quan trọng vì không chỉ xét item đúng có xuất hiện hay không, mà còn xét item đúng xuất hiện ở vị trí nào trong danh sách.

## 6. Kết quả trên full TEST

Các bảng dưới đây chỉ báo cáo kết quả trên full TEST. Giá trị được trích từ output đã lưu trong các notebook.

### 6.1 Recall trên full TEST

| Mô hình | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|
| Popularity Baseline | 0.0115 | 0.0191 | 0.0229 | 0.0611 |
| ItemKNN Baseline | 0.0153 | 0.0191 | 0.0229 | 0.0611 |
| LightFM Baseline | 0.0153 | 0.0267 | 0.0344 | 0.0687 |
| Ablation không Text Encoder | 0.0115 | 0.0191 | 0.0420 | 0.0649 |
| Ablation không KGE | 0.0191 | 0.0229 | 0.0382 | 0.0649 |
| SeRel-LightFM đầy đủ | **0.0267** | **0.0267** | 0.0382 | **0.0725** |

SeRel-LightFM đầy đủ đạt Recall@5 và Recall@50 cao nhất. Ở Recall@10, mô hình đề xuất ngang với LightFM Baseline và cao hơn các baseline còn lại. Ở Recall@20, ablation không Text Encoder đạt giá trị cao nhất, trong khi SeRel-LightFM đầy đủ ngang với ablation không KGE.

### 6.2 NDCG trên full TEST

| Mô hình | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|
| Popularity Baseline | 0.0115 | 0.0141 | 0.0150 | 0.0221 |
| ItemKNN Baseline | 0.0115 | 0.0129 | 0.0139 | 0.0213 |
| LightFM Baseline | 0.0091 | 0.0127 | 0.0146 | 0.0213 |
| Ablation không Text Encoder | 0.0065 | 0.0089 | 0.0147 | 0.0191 |
| Ablation không KGE | 0.0144 | 0.0157 | **0.0196** | 0.0251 |
| SeRel-LightFM đầy đủ | **0.0162** | **0.0162** | 0.0191 | **0.0260** |

SeRel-LightFM đầy đủ đạt NDCG@5, NDCG@10 và NDCG@50 cao nhất. Điều này cho thấy mô hình đề xuất có xu hướng đưa item đúng lên vị trí cao hơn trong danh sách gợi ý, đặc biệt ở các cutoff nhỏ và lớn.

## 7. Phân tích kết quả

### 7.1 So sánh các baseline

Popularity Baseline và ItemKNN Baseline có kết quả khá gần nhau trên full TEST. Popularity nhỉnh hơn ItemKNN ở NDCG@10, NDCG@20 và NDCG@50, cho thấy tín hiệu phổ biến toàn cục là một mốc so sánh đáng kể trong bộ dữ liệu này.

LightFM Baseline cải thiện Recall@10, Recall@20 và Recall@50 so với Popularity và ItemKNN. Tuy nhiên, NDCG của LightFM Baseline chưa vượt trội so với Popularity ở các cutoff được báo cáo. Điều này cho thấy đặc trưng cơ bản giúp mô hình tìm được nhiều item đúng hơn, nhưng chưa luôn đưa item đúng lên vị trí cao nhất.

### 7.2 Vai trò của embedding văn bản

Ablation không KGE vẫn giữ text clusters và đạt NDCG@10 = 0.0157, cao hơn LightFM Baseline và ablation không Text Encoder. Kết quả này cho thấy embedding văn bản là một nguồn tín hiệu quan trọng trong nghiên cứu.

Text Encoder giúp mô hình khai thác ngữ nghĩa từ tiêu đề, thể loại, chủ đề và mô tả nội dung phim. Khi được gom cụm và đưa vào LightFM dưới dạng feature rời rạc, nguồn tín hiệu này cải thiện chất lượng xếp hạng so với việc chỉ sử dụng metadata cơ bản.

### 7.3 Vai trò của KGE

SeRel-LightFM đầy đủ đạt NDCG@10 = 0.0162, cao hơn ablation không KGE với NDCG@10 = 0.0157. Mức cải thiện không lớn nhưng xuất hiện ở các cutoff quan trọng như NDCG@5, NDCG@10 và NDCG@50.

Kết quả này cho thấy KGE có vai trò bổ trợ. Khi kết hợp với embedding văn bản, tín hiệu đồ thị tri thức giúp mô hình cải thiện chất lượng thứ hạng tổng thể trên full TEST.

### 7.4 Mô hình tốt nhất

SeRel-LightFM đầy đủ là mô hình có hiệu năng tổng thể tốt nhất theo NDCG trên full TEST:

- NDCG@5 = 0.0162.
- NDCG@10 = 0.0162.
- NDCG@50 = 0.0260.

Theo Recall, SeRel-LightFM đầy đủ cũng đạt kết quả cao nhất ở Recall@5 và Recall@50:

- Recall@5 = 0.0267.
- Recall@50 = 0.0725.

So với LightFM Baseline, SeRel-LightFM đầy đủ cải thiện:

- Recall@5 từ 0.0153 lên 0.0267.
- Recall@50 từ 0.0687 lên 0.0725.
- NDCG@10 từ 0.0127 lên 0.0162.
- NDCG@50 từ 0.0213 lên 0.0260.

## 8. Kết luận

Nghiên cứu cho thấy SeRel-LightFM là hướng tiếp cận có triển vọng cho bài toán gợi ý phim trong thiết lập phản hồi ẩn. Khi chỉ xét full TEST với hai chỉ số Recall và NDCG, mô hình đề xuất đạt kết quả nổi bật nhất ở NDCG@5, NDCG@10, NDCG@50, Recall@5 và Recall@50.

Kết quả ablation cho thấy embedding văn bản đóng vai trò quan trọng, trong khi KGE bổ sung thêm tín hiệu quan hệ giúp cải thiện chất lượng xếp hạng khi kết hợp với text clusters. Nhìn chung, việc tích hợp metadata, biểu diễn văn bản và biểu diễn đồ thị tri thức vào LightFM giúp nâng cao hiệu quả gợi ý so với các baseline trong nghiên cứu này.

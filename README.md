# Tóm tắt nghiên cứu: Hệ gợi ý phim kết hợp LightFM, biểu diễn văn bản và đồ thị tri thức

## Tóm tắt

Nghiên cứu này xây dựng và đánh giá một hệ gợi ý phim trong bối cảnh phản hồi ẩn (implicit feedback). Rating gốc được chuyển thành tương tác dương khi thỏa điều kiện $r_{u,i} \geq 7$, sau đó các mô hình được so sánh trong cùng một giao thức chia dữ liệu Leave-One-Out.

Mục tiêu chính là đánh giá hiệu quả của mô hình **SeRel-LightFM**, một biến thể mở rộng của LightFM có khả năng kết hợp ba nguồn tín hiệu: metadata phim, embedding văn bản và embedding đồ thị tri thức (Knowledge Graph Embedding, KGE). Kết quả được báo cáo trên **full TEST** và chỉ sử dụng hai nhóm chỉ số: **Recall@K** và **NDCG@K**.

Các mô hình được so sánh gồm:

- **Popularity Baseline:** xếp hạng phim theo số lượng tương tác dương trong tập huấn luyện.
- **ItemKNN Baseline:** lọc cộng tác dựa trên độ tương đồng giữa các item.
- **LightFM Baseline:** LightFM sử dụng đặc trưng item và user cơ bản.
- **Ablation không Text Encoder:** loại bỏ embedding văn bản khỏi SeRel-LightFM.
- **Ablation không KGE:** loại bỏ embedding đồ thị tri thức khỏi SeRel-LightFM.
- **SeRel-LightFM đầy đủ:** kết hợp metadata, embedding văn bản và embedding đồ thị tri thức.

## 1. Bối cảnh và mục tiêu nghiên cứu

Bài toán gợi ý phim cần xếp hạng các item phù hợp nhất cho từng người dùng dựa trên lịch sử tương tác và thông tin bổ trợ. Trong nghiên cứu này, bài toán được tiếp cận theo hướng xếp hạng Top-K, tức mô hình cần đưa item phù hợp lên các vị trí cao trong danh sách đề xuất.

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

Thiết lập chia dữ liệu sử dụng Leave-One-Out (LOO) theo từng người dùng. Quy trình được thực hiện trên **rating gốc trước khi áp dụng threshold**, nhằm bảo toàn thứ tự thời gian và tránh làm thay đổi vị trí tương tác mới nhất của người dùng:

1. Với mỗi người dùng, toàn bộ rating được sắp xếp tăng dần theo thời gian.
2. Rating mới nhất được đưa vào **Test**.
3. Rating ngay trước Test được đưa vào **Validation**.
4. Toàn bộ rating còn lại được đưa vào **Train**.

Sau khi chia LOO xong, từng split mới được chuyển sang phản hồi ẩn bằng ngưỡng:

$$
y_{u,i} =
\begin{cases}
1, & r_{u,i} \geq 7,\\
0, & r_{u,i} < 7.
\end{cases}
$$

Việc tách hai giai đoạn này có ý nghĩa quan trọng:

- **Dataset trước khi lọc threshold** là dữ liệu rating gốc, vẫn giữ đầy đủ mức điểm và thứ tự thời gian. Tập dữ liệu này được dùng để chia LOO, vì mục tiêu của LOO là mô phỏng kịch bản dự đoán tương tác mới nhất của mỗi người dùng dựa trên lịch sử trước đó. Nếu lọc threshold trước khi chia, một số rating thấp có thể bị loại bỏ sớm, làm thay đổi tương tác nào được xem là mới nhất hoặc liền trước mới nhất.
- **Dataset sau khi lọc threshold** là dữ liệu phản hồi ẩn đã được nhị phân hóa. Tập dữ liệu này được dùng để huấn luyện và đánh giá mô hình gợi ý, trong đó rating thỏa $r_{u,i} \geq 7$ được xem là tín hiệu người dùng có xu hướng quan tâm đến item. Các rating thấp hơn ngưỡng không được xem là ground truth dương trong đánh giá Top-K.

Trong quá trình huấn luyện và đánh giá, chỉ các tương tác dương ($y_{u,i}=1$) được xem là ground truth phù hợp. Vì vậy, một số người dùng vẫn có rating trong Test nhưng không được tính vào đánh giá nếu rating đó nhỏ hơn 7. Sau bước threshold, full TEST có 262 người dùng được dùng trong đánh giá.

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

Trong đó $N(j,K)$ là tập K item lân cận của item $j$, còn $I_u^+$ là tập item dương trong lịch sử huấn luyện của user $u$.

### 3.3 LightFM Baseline

LightFM Baseline sử dụng hàm mất mát WARP để học mô hình xếp hạng. Mô hình kết hợp tương tác user-item với đặc trưng phụ:

- Đặc trưng item: `genre`, `topic`, `country`, `year`, `duration`.
- Đặc trưng user: các cờ hành vi, nhóm giờ ưa thích, năm tạo tài khoản và ngày trong tuần.

Baseline này chưa sử dụng embedding văn bản hoặc embedding đồ thị tri thức. Do đó, nó đóng vai trò mốc so sánh chính cho SeRel-LightFM.

### 3.4 SeRel-LightFM

SeRel-LightFM mở rộng LightFM bằng cách bổ sung các cụm đặc trưng được tạo từ embedding văn bản và embedding đồ thị tri thức. Ý tưởng chính là không đưa trực tiếp vector dense của Sentence Transformer hoặc TransE vào LightFM, mà chuyển chúng thành các feature rời rạc có trọng số thông qua KMeans và soft assignment. Cách làm này giữ được giao diện feature-based của LightFM, đồng thời đưa thêm tín hiệu ngữ nghĩa và quan hệ vào cùng không gian xếp hạng WARP.

| Nhóm đặc trưng | Nguồn tín hiệu | Cách đưa vào LightFM |
|---|---|---|
| Metadata item - CFS | Thể loại, chủ đề, quốc gia, năm phát hành, thời lượng | Feature rời rạc và biến liên tục đã chuẩn hóa |
| Hồ sơ user - CFS | Thông tin tĩnh và hành vi tổng quát | Feature rời rạc phía user |
| Embedding văn bản - ARSR | Sentence Transformer trên mô tả đa trường của phim | `text_cluster:*` trong item features |
| Embedding đồ thị tri thức phía item - ARSR | TransE trên KG mở rộng | `kg_cluster:*` trong item features |
| Embedding đồ thị tri thức phía user - ARSR | TransE trên user nodes trong KG | `user_kg_cluster:*` trong user features |

#### Biểu diễn văn bản

Pipeline Text Encode được thực hiện trên toàn bộ `movies_df` sau khi metadata đã được làm sạch, không chỉ trên phim xuất hiện trong `train_df`. Bước này không dùng rating, không dùng threshold và không dùng nhãn validation/test; nó chỉ học biểu diễn từ nội dung item.

Với mỗi phim, notebook tạo một chuỗi văn bản đa trường bằng hàm `build_text_input(row)`. Các thành phần được nối lại gồm:

- `original_title`, nếu tiêu đề không rỗng.
- `genres`, được gắn nhãn `Generos:`.
- `topics`, được gắn nhãn `Temas:` và bỏ qua giá trị rỗng hoặc `"nan"`.
- `directors`, được gắn nhãn `Directores:`.
- Tối đa 5 diễn viên đầu tiên từ `actors`, được gắn nhãn `Reparto:`.
- `plot_clean`, tức nội dung mô tả đã qua bước sửa byte-string và lỗi encoding.

Nếu phim không có metadata văn bản hợp lệ, chuỗi đầu vào được đặt là `"(no metadata)"`. Toàn bộ chuỗi được mã hóa bằng Sentence Transformer `paraphrase-multilingual-MiniLM-L12-v2`. Notebook bật `normalize_embeddings=True`, nên mỗi embedding đầu ra đã được chuẩn hóa $L_2$; khi đó cosine similarity giữa embedding và centroid có thể tính bằng dot product.

Sau khi có `text_embeddings`, pipeline gom cụm bằng KMeans với `K_TEXT_CLUSTERS = 200`, `random_state = 42` và `n_init = 10`. Centroid của KMeans cũng được chuẩn hóa $L_2$. Với mỗi phim, pipeline tính similarity tới toàn bộ centroid, lấy `TEXT_TOP_K = 3` cụm gần nhất, sắp xếp theo similarity giảm dần, rồi gán trọng số bằng softmax có temperature `TEXT_TEMP = 0.1`:

$$
w_c =
\frac{\exp(s_c / T)}
{\sum_{c'} \exp(s_{c'} / T)},
$$

trong đó $s_c$ là cosine similarity giữa embedding của phim và centroid cụm $c$, còn $T$ là temperature. Kết quả cuối của bước này là `text_cluster_assignments`, trong đó mỗi phim có danh sách ba cặp `(cluster_id, weight)`. Các cặp này được inject vào `item_feat_dicts` dưới dạng feature `text_cluster:{cluster_id}` với trọng số tương ứng.

Như vậy, Text Encode biến nội dung tự do của phim thành một tập feature thưa có trọng số. LightFM không nhìn thấy trực tiếp câu mô tả hoặc vector Sentence Transformer; nó chỉ nhận các cụm ngữ nghĩa gần nhất của item như feature phụ khi học xếp hạng.

#### Biểu diễn đồ thị tri thức

Pipeline KGE xây dựng một đồ thị tri thức mở rộng gồm movie entities, user entities và attribute entities, sau đó dùng TransE để học embedding chung cho các entity này. Cần phân biệt rõ dữ liệu dùng cho KGE với dữ liệu sau khi lọc threshold của mô hình gợi ý:

- **KGE không dùng `valid_df` hoặc `test_df`** để tạo cạnh user-item, nhằm tránh rò rỉ dữ liệu đánh giá.
- **KGE dùng `train_df` sau khi chia LOO nhưng trước khi loại bỏ rating dưới threshold.** Nói cách khác, `train_df` vẫn là rating gốc, gồm cả rating cao và rating thấp.
- Threshold $r_{u,i} \geq 7$ chỉ được dùng bên trong `train_df` để gán loại quan hệ:
  - Rating thỏa threshold tạo cạnh `user_likes`.
  - Rating thấp hơn threshold có thể tạo cạnh `user_watched` nếu cấu hình `USE_USER_WATCHED_REL=True`.
- Các quan hệ item-attribute được tạo từ metadata của toàn bộ `movies_df`, không phụ thuộc vào rating hoặc threshold.

Phần item-attribute của KG được tạo từ 9 nhóm metadata trong `movies_df`:

| Cột metadata | Quan hệ trong KG |
|---|---|
| `genres` | `has_genre` |
| `directors` | `directed_by` |
| `actors` | `acted_by` |
| `country_name` | `from_country` |
| `topics` | `has_topic` |
| `script` | `written_by` |
| `producer` | `produced_by` |
| `music` | `scored_by` |
| `photography` | `shot_by` |

Mỗi phim được biểu diễn bằng entity `movie:{movie_id}`. Mỗi token metadata tạo một attribute entity theo dạng `{column}:{token}`. Ví dụ, một thể loại trong cột `genres` tạo triple `(movie:id, has_genre, genres:token)`. Các token rỗng hoặc `"nan"` bị bỏ qua.

Phần user-item của KG chỉ đến từ raw `train_df`. Với mỗi dòng train, user được biểu diễn bằng entity `user:{id}` và item bằng `movie:{movie_id}`. Nếu `rate >= POSITIVE_THRESHOLD`, pipeline tạo triple `(user:id, user_likes, movie:id)`. Nếu `rate < POSITIVE_THRESHOLD` và `USE_USER_WATCHED_REL=True`, pipeline tạo triple `(user:id, user_watched, movie:id)`. Vì vậy, threshold trong KGE không phải là bước lọc bỏ toàn bộ rating thấp; nó chỉ quyết định loại quan hệ giữa user và movie.

Sau khi gộp item-attribute triples và user-item triples, notebook tạo `TriplesFactory` của PyKEEN rồi chia nội bộ KG theo tỉ lệ 90/10 thành `training_tf` và `testing_tf`. Split này là split riêng cho huấn luyện TransE, khác với `valid_df` và `test_df` của bài toán recommendation.

TransE được huấn luyện với embedding dimension `KGE_DIM = 64`, `KGE_EPOCHS = 100`, optimizer Adam và learning rate `1e-3`. Mô hình học embedding sao cho mỗi bộ ba $(h,r,t)$ thỏa gần đúng:

$$
\mathbf{h} + \mathbf{r} \approx \mathbf{t}.
$$

Sau khi huấn luyện, notebook trích xuất embedding cho toàn bộ entity trong KG, rồi map về hai ma trận:

- `kg_embeddings`: embedding của toàn bộ phim trong `movies_df`, theo thứ tự movie dataframe.
- `user_kg_matrix_normed`: embedding của toàn bộ user trong `users_df`, theo thứ tự user dataframe.

Entity movie hoặc user không xuất hiện trong KG được gán vector 0 trước khi chuẩn hóa an toàn. Các vector còn lại được chuẩn hóa $L_2$ để phục vụ gom cụm bằng cosine similarity.

KGE phía item và KGE phía user được xử lý riêng:

- **Item KGE clusters:** KMeans trên `kg_embeddings` với `K_KG_CLUSTERS = 150`. Với mỗi phim, lấy `KG_TOP_K = 3` cụm gần nhất, tính trọng số bằng softmax với `KG_TEMP = 0.1`, rồi inject vào `item_feat_dicts` dưới dạng `kg_cluster:{cluster_id}`.
- **User KGE clusters:** KMeans trên `user_kg_matrix_normed` với `K_USER_KG_CLUSTERS = 30`. Với mỗi user, lấy `USER_KG_TOP_K = 3` cụm gần nhất, tính trọng số bằng softmax với `USER_KG_TEMP = 0.1`, rồi inject vào `user_feat_dicts` dưới dạng `user_kg_cluster:{cluster_id}`.

Kết quả là LightFM nhận đồng thời ba lớp tín hiệu: Tier 1 metadata item, Tier 2 text/item-KGE/user-KGE clusters và Tier 3 user behavioral features. Trong mô hình đầy đủ, item features gồm các feature nội dung cơ bản, `text_cluster:*` và `kg_cluster:*`; user features gồm các feature hành vi cơ bản và `user_kg_cluster:*`.

## 4. Thiết kế ablation

Hai biến thể ablation được dùng để phân tích đóng góp của từng thành phần trong SeRel-LightFM.

### 4.1 Ablation không KGE

Biến thể này loại bỏ toàn bộ nhánh đồ thị tri thức, gồm xây dựng KG, huấn luyện TransE và tạo KGE clusters. Mô hình vẫn giữ metadata, user features và text clusters. Mục tiêu là đo tác động của KGE khi embedding văn bản vẫn được sử dụng.

### 4.2 Ablation không Text Encoder

Biến thể này loại bỏ Sentence Transformer và text clusters. Mô hình vẫn giữ metadata, user features và KGE clusters. Mục tiêu là đo tác động của embedding văn bản khi tín hiệu đồ thị tri thức vẫn được sử dụng.

## 5. Chỉ số đánh giá

Nghiên cứu chỉ sử dụng hai chỉ số trên full TEST:

### 5.1 Recall@K

Recall@K đo tỷ lệ item đúng được tìm thấy trong top-K gợi ý:

$$
\mathrm{Recall@K} =
\frac{|G_u \cap P_u^K|}{|G_u|},
$$

trong đó $G_u$ là tập item đúng của user $u$, còn $P_u^K$ là tập top-K item được mô hình đề xuất.

### 5.2 NDCG@K

NDCG@K đánh giá chất lượng thứ hạng của item đúng trong top-K. Chỉ số này ưu tiên các mô hình đưa item đúng lên vị trí cao hơn:

$$
\mathrm{NDCG@K} =
\frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}.
$$

Với bài toán gợi ý Top-K, NDCG thường quan trọng vì không chỉ xét item đúng có xuất hiện hay không, mà còn xét item đúng xuất hiện ở vị trí nào trong danh sách.

## 6. Kết quả trên full TEST

Các bảng dưới đây chỉ báo cáo kết quả trên full TEST. Giá trị được trích từ output đã lưu trong các notebook.

### 6.1 Recall trên full TEST: so sánh các phương pháp chính

| Mô hình | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|
| Popularity Baseline | 0.0115 | 0.0191 | 0.0229 | 0.0611 |
| ItemKNN Baseline | 0.0153 | 0.0191 | 0.0229 | 0.0611 |
| LightFM Baseline | 0.0153 | 0.0267 | 0.0344 | 0.0687 |
| SeRel-LightFM đầy đủ | **0.0267** | **0.0267** | 0.0382 | **0.0725** |

SeRel-LightFM đầy đủ đạt Recall@5 và Recall@50 cao nhất trong nhóm phương pháp chính. Ở Recall@10, mô hình đề xuất ngang với LightFM Baseline và cao hơn Popularity Baseline, ItemKNN Baseline.

### 6.2 NDCG trên full TEST: so sánh các phương pháp chính

| Mô hình | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|
| Popularity Baseline | 0.0115 | 0.0141 | 0.0150 | 0.0221 |
| ItemKNN Baseline | 0.0115 | 0.0129 | 0.0139 | 0.0213 |
| LightFM Baseline | 0.0091 | 0.0127 | 0.0146 | 0.0213 |
| SeRel-LightFM đầy đủ | **0.0162** | **0.0162** | 0.0191 | **0.0260** |

SeRel-LightFM đầy đủ đạt NDCG@5, NDCG@10 và NDCG@50 cao nhất. Điều này cho thấy mô hình đề xuất có xu hướng đưa item đúng lên vị trí cao hơn trong danh sách gợi ý, đặc biệt ở các cutoff nhỏ và lớn.

### 6.3 Recall trên full TEST: phân tích ablation

| Mô hình | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|
| LightFM Baseline | 0.0153 | **0.0267** | 0.0344 | 0.0687 |
| Ablation không Text Encoder | 0.0115 | 0.0191 | **0.0420** | 0.0649 |
| Ablation không KGE | 0.0191 | 0.0229 | 0.0382 | 0.0649 |
| SeRel-LightFM đầy đủ | **0.0267** | **0.0267** | 0.0382 | **0.0725** |

Bảng ablation cho thấy SeRel-LightFM đầy đủ đạt Recall@5 và Recall@50 cao nhất. Ở Recall@20, biến thể không Text Encoder đạt giá trị cao nhất, cho thấy đóng góp của từng thành phần có thể khác nhau theo cutoff.

### 6.4 NDCG trên full TEST: phân tích ablation

| Mô hình | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|
| LightFM Baseline | 0.0091 | 0.0127 | 0.0146 | 0.0213 |
| Ablation không Text Encoder | 0.0065 | 0.0089 | 0.0147 | 0.0191 |
| Ablation không KGE | 0.0144 | 0.0157 | **0.0196** | 0.0251 |
| SeRel-LightFM đầy đủ | **0.0162** | **0.0162** | 0.0191 | **0.0260** |

Ở nhóm ablation, SeRel-LightFM đầy đủ đạt NDCG@5, NDCG@10 và NDCG@50 cao nhất. Ablation không KGE đạt NDCG@20 cao nhất, cho thấy text clusters đóng góp mạnh ở cutoff trung bình, trong khi KGE giúp cải thiện thêm khi kết hợp trong mô hình đầy đủ.

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

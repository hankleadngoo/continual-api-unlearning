# Báo cáo đánh giá phương pháp steering bằng vector trung bình và MLP gate

Ngày tổng hợp: 27/09/2026.

## 1. Kết luận chính

Thí nghiệm được cung cấp **chưa chứng minh phương pháp thay thế deprecated API bằng replacement API hiệu quả**. Trên 200 mẫu U_dep, số mẫu được đánh dấu có deprecated API giảm từ 36 xuống 34, nhưng không có mẫu replacement sạch theo các cờ đã lưu; số mẫu không được phát hiện có cả hai loại API tăng từ 164 lên 166. Trên 200 mẫu U_nondep, các tổng số không thay đổi.

Đồng thời, gate gần như luôn trả a ≈ 1, và bộ đánh giá có lỗi nhận diện API. Do đó, không nên kết luận rằng toàn bộ hướng activation steering không khả thi. Kết luận được dữ liệu hỗ trợ là: **cấu hình và phép đo hiện tại chưa tạo ra bằng chứng cải thiện đáng tin cậy**.

Các số liệu dưới đây thuộc repo của người bạn, thư mục `results/L12_t1.0_neg`, đã được đọc trong phiên phân tích này. Chúng không phải kết quả chạy pipeline t=0.5, 500 mẫu trong project hiện tại. Chưa có số liệu pretrained của cấu hình mới trong phạm vi báo cáo này.

## 2. Bài toán và mục tiêu

Đầu vào x là đoạn code `probing input`. Model cần sinh phần tiếp theo, sử dụng replacement API thay cho deprecated API khi phù hợp. Thành công không chỉ là không sinh deprecated API: sinh rỗng, bỏ qua thao tác cần thiết hoặc sinh code không liên quan cũng có thể tránh được API cũ.

Mục tiêu đánh giá cần bao gồm:

- Giảm mẫu sinh deprecated API.
- Tăng mẫu sinh replacement API mà không còn deprecated API.
- Hạn chế mismatch và suy giảm trên mẫu vốn đã xử lý đúng.
- Nếu muốn kết luận về chất lượng chương trình, kiểm tra thêm cú pháp, tham số, hành vi và functional tests.

Việc đóng băng model và sửa activation là một can thiệp có thể bật/tắt. Chỉ từ thí nghiệm này chưa thể khẳng định kiến thức về API cũ đã bị xóa khỏi trọng số, hoặc đã đạt machine unlearning theo nghĩa mạnh.

## 3. Phương pháp

### 3.1. Tạo vector steering

Với mỗi cặp trong D_forget, model gốc đóng băng trích xuất hidden state của token cuối tại đầu ra block L:

```text
h_neg[i] = h_L(prompt(x_i) + y_neg[i])
h_pos[i] = h_L(prompt(x_i) + y_pos[i])

v_dep = mean_i(h_neg[i])
v_rep = mean_i(h_pos[i])
v_steer = v_rep - v_dep
```

Vector này là chênh lệch trung bình của hai tập activation. Cách gọi “trừ khái niệm cũ, cộng khái niệm mới” là trực giác thiết kế, không phải tính chất đã được chứng minh. Hidden state còn chứa thông tin về ngữ cảnh, cú pháp và token; hiệu hai trung bình không được đảm bảo là một hướng chỉ mã hóa trạng thái deprecated/replacement.

Prompt của thí nghiệm:

````text
Complete and output the next line for the following Python function:
```python
{code_context}
```
````

### 3.2. Học MLP gate

Gate có một lớp ẩn rộng 256 và sigmoid đầu ra:

```text
a = sigmoid(MLP(h_in))
loss = mean_i[1 - cos(h_neg[i] + a[i] * v_steer, h_pos[i])]
```

Trong run được báo cáo, `gate_train_input=neg`, nên `h_in=h_neg`. Mục tiêu là giảm khoảng cách góc sau khi can thiệp. Đây **không phải supervised classification về nguy cơ sinh deprecated API**: không có nhãn phân loại trong loss này. Giá trị sigmoid nằm trong [0,1] nhưng chưa được hiệu chuẩn thành xác suất.

### 3.3. Inference

Model chỉ nhận prompt của mẫu test, không biết y_pos hoặc y_neg:

```text
a = sigmoid(MLP(h_L(prompt(x))))
h' = h + t * a * v_steer
```

Vector và trọng số gate đã được tính ở training. Repo của người bạn lấy gate từ token cuối prompt rồi broadcast delta lên mọi vị trí trong lượt prefill. Hook không can thiệp vào các bước decode có độ dài một token.

Pipeline hiện tại của project khác ở hai điểm: gate học từ hidden cuối prompt và hook tiếp tục áp dụng gate theo từng token decode. Vì vậy không được dùng kết quả của repo bạn ấy như kết quả trực tiếp của code hiện tại.

## 4. Thiết lập thí nghiệm được cung cấp

| Thành phần | Giá trị |
|---|---|
| Model | deepseek-ai/deepseek-coder-1.3b-instruct |
| Block | Chỉ số 12 |
| Strength | t=1.0 |
| Gate training input | h(x+y_neg) |
| D_forget | Không đặt giới hạn n_forget |
| Test | 200 mẫu U_dep và 200 mẫu U_nondep |
| Context / generation | Tối đa 1024 token đầu vào / 48 token sinh |
| Batch extraction / generation / gate | 32 / 32 / 64 |
| Gate hidden size | 256 |
| Training | 10 epoch, learning rate 0.001 |
| Seed | 42 |

Đây là một cấu hình, một seed và một tập mẫu giới hạn. Chưa có lặp thí nghiệm, khoảng tin cậy hoặc khảo sát layer/strength để kết luận về toàn bộ không gian cấu hình.

## 5. Kết quả

### 5.1. Số liệu tổng hợp trong results.json

| Tập | Chế độ | N | repAPI | depAPI |
|---|---|---:|---:|---:|
| U_dep | Baseline | 200 | 18 (9%) | 36 (18%) |
| U_dep | Steered | 200 | 16 (8%) | 34 (17%) |
| U_nondep | Baseline | 200 | 7 (3,5%) | 1 (0,5%) |
| U_nondep | Steered | 200 | 7 (3,5%) | 1 (0,5%) |

Hai cột repAPI và depAPI không loại trừ nhau. Chúng là số mẫu được bộ dò đánh dấu, không phải số lần gọi API và không phải số chương trình đúng.

### 5.2. Tính lại các nhóm loại trừ nhau từ gate_scores.json

| Tập | Chế độ | Chỉ dep | Cả dep và rep | Chỉ rep | Không cả hai |
|---|---|---:|---:|---:|---:|
| U_dep | Baseline | 18 | 18 | 0 | 164 |
| U_dep | Steered | 18 | 16 | 0 | 166 |
| U_nondep | Baseline | 1 | 0 | 7 | 192 |
| U_nondep | Steered | 1 | 0 | 7 | 192 |

Các nhóm này được tính từ cờ đã lưu, **chưa sửa lỗi bộ dò**. Tổng hai tập: số mẫu có dep giảm 37→35; chỉ rep giữ ở 7; không cả hai tăng 356→358. Không có bằng chứng tăng replacement sạch theo phép đo này.

### 5.3. Chuyển đổi từng mẫu

Trên U_dep, chỉ ba mẫu đổi nhóm: một mẫu cả hai→chỉ dep, một mẫu cả hai→mismatch, một mẫu chỉ dep→mismatch. Không có chuyển đổi sang chỉ rep.

Trên U_nondep, một mẫu mismatch→rep và một mẫu rep→mismatch; các mẫu còn lại giữ nhóm. Tổng số không đổi không có nghĩa output từng mẫu không thay đổi.

### 5.4. Gate scores

| Thống kê a trên steered | U_dep | U_nondep |
|---|---:|---:|
| Trung bình | 0.999899 | 0.999615 |
| Nhỏ nhất | 0.998538 | 0.989342 |
| Trung vị | 0.999986 | 0.999926 |
| a > 0.99 | 200/200 | 199/200 |
| a > 0.999 | 199/200 | 185/200 |

Gate gần như luôn bật trên các prompt được kiểm tra. Với t=1, can thiệp gần tương đương cộng một vector cố định. Điều này không chứng minh gate đã nhận biết chính xác nguy cơ deprecated; ngược lại, nó cho thấy khả năng điều chỉnh cường độ theo mẫu rất hạn chế trong run này.

## 6. Các vấn đề và nguyên nhân có thể khiến phương pháp chưa hiệu quả

### 6.1. Sai lệch phép đo: đã xác nhận trong code và output

Bộ dò dùng regex hậu tố tên API và quét trực tiếp văn bản sinh, không loại comment/string. Hai hệ quả quan sát được:

- Mẫu U_dep số 99 sinh `torch.qr(A)` nhưng có rep=True dù replacement mong muốn là `torch.linalg.qr`; hậu tố `qr` không phân biệt được hai lời gọi.
- Mẫu U_dep số 74 có API trong comment vẫn được tính.

Vì vậy, cần chấm lại trước khi xem các số đếm là chất lượng API thực tế. Không thể sửa kết quả chỉ bằng cách lấy repAPI trừ depAPI.

### 6.2. Gate tối ưu alignment, không học quyết định có nên can thiệp

Loss chỉ yêu cầu đưa h_neg gần h_pos theo cosine. Không có thành phần trực tiếp yêu cầu gate tắt khi model đã sinh API đúng. Nếu tăng a giúp loss giảm trên nhiều cặp, việc a tiến về 1 là nghiệm mà mục tiêu hiện tại cho phép.

**Bằng chứng:** a gần 1 trên cả hai tập. **Giới hạn:** chưa có activation/loss theo từng mẫu để chứng minh bão hòa là nguyên nhân duy nhất của kết quả kém.

### 6.3. Lệch đầu vào training–inference

Gate học từ h(x+y_neg), trạng thái đã thấy completion, nhưng infer nhận h(x), trạng thái trước khi sinh completion. Đây là khác biệt được xác nhận từ cấu hình và code. Nó có thể làm gate không tổng quát, nhưng cần so sánh có kiểm soát với `gate_train_input=probe` để đo mức ảnh hưởng.

### 6.4. Cosine loss là mục tiêu thay thế, không bảo đảm sinh đúng API

Gần nhau về góc tại một block không đồng nghĩa các block tiếp theo sẽ tăng xác suất đúng chuỗi replacement. Loss cũng không kiểm tra tham số API, cú pháp hoặc chức năng. Ngoài ra, cosine không trực tiếp ràng buộc norm của hidden state sau can thiệp.

**Bằng chứng về giới hạn kết quả:** giảm dep không đi cùng tăng chỉ rep. **Chưa biết:** cần đo xác suất completion và chất lượng output để quy kết thất bại cho khoảng cách giữa loss và mục tiêu sinh.

### 6.5. Một vector trung bình có thể không đủ cho nhiều API

Các phép thay API khác nhau có thể đòi hỏi thay tên, tham số hoặc cấu trúc code khác nhau. Trung bình chung có thể làm các hướng triệt tiêu hoặc giữ lại đặc trưng không liên quan. Đây là giả thuyết cấu trúc, chưa được kiểm chứng bằng ablation trong run này.

### 6.6. Vị trí và cường độ can thiệp chưa được kiểm chứng

Vector học từ token cuối completion nhưng được broadcast lên tất cả vị trí prompt. Cùng t=1 không đồng nghĩa cùng mức nhiễu giữa các model/layer vì norm vector khác nhau. Cần đo `||t*a*v|| / ||h||` thay vì chỉ nhìn t.

Chưa có dữ liệu để khẳng định t=1 quá lớn. Giảm t có thể giảm tác hại nhưng không tự giải quyết gate luôn bật hoặc vector sai hướng.

### 6.7. Baseline có tỷ lệ phát hiện API thấp

Phần lớn mẫu thuộc nhóm không cả hai ngay cả trước steering. Cần kiểm tra prompt, vị trí dòng cần hoàn thành, truncation, 48 token sinh có đủ không, và output có lặp prompt hoặc chuyển sang giải thích không. Tỷ lệ mismatch cao không tự chứng minh model yếu: nó cũng có thể phản ánh cách đặt bài toán hoặc phép đo.

## 7. Hướng kiểm chứng tiếp theo

1. **Sửa phép đo trước:** phân tích token code, bỏ comment/string, ánh xạ alias chính xác; tách chỉ dep/cả hai/chỉ rep/mismatch. Chấm lại output đã có trước khi sinh lại.
2. **So sánh cố định gate:** baseline; a=1; gate hiện tại, cùng vector/layer/t/prompt/mẫu. Nếu gate và a=1 gần như giống nhau thì gate chưa mang lại lợi ích chọn lọc.
3. **Ablation đầu vào gate:** h_neg so với h_prompt, giữ các yếu tố khác cố định. Không đánh đồng U_dep/U_nondep với nhãn nguy cơ thực tế của một model/prompt mới.
4. **Nếu cần classifier nguy cơ:** tạo nhãn từ baseline generation trên tập training riêng. Có dep là positive; replacement sạch là negative; mismatch cần xử lý riêng. Học BCE từ h_prompt và đánh giá false-positive/false-negative trên validation. Kiểm tra riêng lợi ích của cosine loss nếu kết hợp.
5. **Giữ nhất quán vị trí gate:** gate học trên prompt thì kiểm tra phương án tính a ở prompt và giữ trong decode; gate theo từng token cần được kiểm chứng với đầu vào tương ứng.
6. **Ablation vector và hook:** vector toàn cục so với theo nhóm API; last-prompt-token so với broadcast prefill hoặc decode. Mỗi lần chỉ đổi một yếu tố.
7. **Đánh giá cuối trên test độc lập:** chọn cấu hình bằng validation. Ghi chuyển đổi từng mẫu, chi phí thời gian, và functional tests nếu có. 500 mẫu chỉ là kiểm tra nhanh, chưa đại diện toàn bộ benchmark.

Không thay đổi tất cả thành phần cùng lúc vì sẽ không xác định được yếu tố nào thực sự tạo cải thiện.

## 8. Quan hệ với code project hiện tại

Project đang dùng DeepSeek cùng next-line instruction, gate học từ h_prompt, cố định t=0.5, tối đa 500 mẫu test tổng cộng, generation theo batch mặc định 8. Steering tiếp tục ở decode; log hook chỉ ghi prefill để giảm overhead. Training vẫn dùng toàn bộ cặp D_forget hợp lệ theo mặc định.

Các thay đổi này giúp đồng nhất đầu vào gate ban đầu và giảm chi phí đánh giá. **Chúng chưa sửa bản chất cosine-only gate không có nhãn phân biệt nguy cơ**, và chưa có kết quả pretrained trong báo cáo này để khẳng định cải thiện. Kiểm thử offline xác nhận cơ chế phần mềm, không chứng minh hiệu quả unlearning.

## 9. Nguồn và phạm vi tái lập

- [results.json: cấu hình và tổng số](https://github.com/dinhngoctuyen4125/MLLMEraser/blob/main/results/L12_t1.0_neg/results.json)
- [gate_scores.json: hệ số gate, cờ rep/dep](https://github.com/dinhngoctuyen4125/MLLMEraser/blob/main/results/L12_t1.0_neg/gate_scores.json)
- [samples.json: đầu vào và output để kiểm tra định tính](https://github.com/dinhngoctuyen4125/MLLMEraser/blob/main/results/L12_t1.0_neg/samples.json)
- [algo.py: objective, hook và regex đánh giá](https://github.com/dinhngoctuyen4125/MLLMEraser/blob/main/algo.py)

Các thống kê gate, nhóm loại trừ nhau và chuyển đổi được tính lại từ JSON đã đọc, không phải lần chạy model mới. Các link trỏ tới nhánh main có thể thay đổi; chưa có commit SHA của run được ghi nhận trong báo cáo. Chưa có trọng số gate/vector hay log loss theo epoch để kiểm tra lại quá trình hội tụ.

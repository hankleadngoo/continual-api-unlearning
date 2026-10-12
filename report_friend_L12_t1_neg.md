# Báo cáo lần chạy L12_t1.0_neg — repo MLLMEraser của dinhngoctuyen4125

Ngày phân tích: 28/09/2026.

## 1. Phạm vi và kết luận

Báo cáo này chỉ phân tích thí nghiệm `results/L12_t1.0_neg` trong repo `dinhngoctuyen4125/MLLMEraser`. Không sử dụng kết quả hay cơ chế triển khai của project khác.

**Kết luận: run này chưa cho thấy phương pháp đạt mục tiêu chuyển deprecated API sang replacement API.** Số mẫu có deprecated giảm nhẹ nhưng replacement sạch không tăng; gate gần như luôn mở. Bộ chấm API còn có lỗi nên cần chấm lại output trước khi kết luận về chất lượng thực tế. Không đủ bằng chứng để khẳng định mọi cấu hình activation steering đều thất bại.

Code và ba file JSON được kiểm tra tại commit `b0ad8632ed3dc71d95234b204301964e2217b113`. Đây là commit được đọc khi lập báo cáo, **không phải commit tạo run đã được xác nhận**: `results.json` không lưu run commit. Cấu hình thí nghiệm được xác nhận từ `args`; mô tả triển khai được đối chiếu với code tại commit trên. Nếu code đã đổi sau khi chạy, cần lịch sử hoặc log chạy để khôi phục chính xác phiên bản thực thi.

## 2. Bài toán

Với đoạn code đầu vào x (`probing input`), model cần sinh dòng tiếp theo sử dụng replacement API thích hợp, tránh deprecated API. Một kết quả chỉ không chứa API cũ chưa đủ thành công: nó có thể là code không liên quan, bỏ mất phép tính hoặc văn bản giải thích.

Hai tập test được đọc độc lập:

- `D_test_U_dep.json`: nhóm được benchmark xác định liên quan tới lỗi sinh deprecated trong thiết lập gốc.
- `D_test_U_nondep.json`: nhóm còn lại của benchmark.

Nhãn tập không bảo đảm hành vi trong lần chạy này. Ví dụ, baseline trên 200 mẫu U_dep chỉ được bộ chấm đánh dấu có dep ở 36 mẫu.

## 3. Cấu hình đã lưu

| Tham số | Giá trị và ý nghĩa |
|---|---|
| Model | `deepseek-ai/deepseek-coder-1.3b-instruct` |
| Layer | Chỉ số block 12, đánh số từ 0 |
| Strength | `t = 1.0` |
| Gate training input | `neg`: hidden state của prompt nối y_neg |
| n_forget | `null`: không giới hạn số dòng khi đọc D_forget |
| max_test | 200 mẫu mỗi file test |
| max_len | 1024 token khi tokenize |
| max_new_tokens | Tối đa 48 token sinh cho mỗi mẫu |
| extract_bs | Batch trích xuất hidden: 32 |
| gen_bs | Batch inference: 32 |
| gate_bs | Batch học gate: 64 |
| gate_hidden | Lớp ẩn MLP: 256 |
| epochs / lr | 10 epoch / 0.001 |
| n_dump | Tối đa 200 mẫu chi tiết mỗi nhóm kết quả |
| seed | 42 |

`n_forget=null` không cho biết số cặp training thực tế. Ba file kết quả không ghi số đó, thời gian chạy, GPU hay phiên bản thư viện. Không nên suy ra run đã dùng đúng 9.667 cặp.

Theo code được kiểm tra, model được load bfloat16, đặt eval; trích xuất bằng no_grad và optimizer chỉ cập nhật MLP. Tokenizer padding/truncation bên trái, pad token dùng EOS nếu chưa có. Code đọc `data[:limit]`: **200 mẫu đầu mỗi file**, không phải lấy ngẫu nhiên với seed 42.

## 4. Pipeline thực hiện

### 4.1. Tạo prompt

Code `build_prompt` tại commit được kiểm tra tạo chuỗi:

```python
prompt_text = (
    "Complete and output the next line for the following Python function:\n"
    "```python\n"
    f"{code_context}"
)
```

Không có dấu đóng khối Markdown ở cuối, không áp dụng chat template trong hàm này. Đây là khác biệt với mô tả prompt có dấu đóng trong trao đổi trước. Artifact không lưu đầy đủ formatted prompt hoặc commit chạy nên chưa xác nhận được run cũ có dùng đúng phiên bản hàm hiện tại hay không.

Chỉ `probing input` được đưa vào prompt test. Nhãn replacement/deprecated và hàm tham chiếu dùng để chấm/ghi báo cáo, không cung cấp trước đáp án cho model.

### 4.2. Trích xuất hidden và tính vector

Với mỗi cặp D_forget:

```text
p_i = build_prompt(x_i)
h_neg[i] = last_token_hidden_L(p_i + y_neg[i])
h_pos[i] = last_token_hidden_L(p_i + y_pos[i])

v_dep = mean(h_neg)
v_rep = mean(h_pos)
v_steer = v_rep - v_dep
```

Code lấy `hidden_states[layer+1][:, -1, :]`, tương ứng đầu ra block 12 trong trường hợp này, theo batch 32. Prompt và completion được nối thành chuỗi trước khi tokenize. Hai nhánh được tokenize/truncate riêng; với chuỗi quá dài, phần prompt được giữ lại có thể khác nhau khi độ dài hai completion khác nhau.

Vector không được chuẩn hóa trong hàm tạo vector. “Trừ khái niệm cũ, cộng khái niệm mới” là trực giác; hiệu trung bình không được bảo đảm chỉ biểu diễn sự khác biệt API.

### 4.3. Huấn luyện gate

```text
MLP: Linear(H,256) → ReLU → Linear(256,1) → sigmoid
h_in = h_neg                        # cấu hình neg
a_i = sigmoid(MLP(h_in[i]))
loss = mean(1 - cos(h_neg[i] + a_i*v_steer, h_pos[i]))
```

Gate học 10 epoch bằng AdamW, lr=0.001, minibatch 64 có shuffle. Chỉ trọng số gate được tối ưu; vector và hidden đã trích xuất giữ cố định.

Loss này không có nhãn “sẽ sinh dep / không sinh dep”, không có BCE, cũng không trực tiếp tối ưu xác suất token replacement. Do đó a là hệ số steering, chưa phải xác suất phân loại nguy cơ được hiệu chuẩn. Log code có loss trung bình và mean a mỗi epoch, nhưng ba JSON được cung cấp không chứa lịch sử đó.

### 4.4. Inference với steering

Với prompt test, tại block 12:

```text
a = sigmoid(MLP(h_last_prompt))
delta = t * a * v_steer
h'[position] = h[position] + delta
```

Code broadcast cùng delta lên toàn bộ vị trí ở prefill. Hook chỉ chạy khi chiều sequence > 1; khi cached decode xử lý một token, hook không cộng thêm vector. Can thiệp prefill vẫn có thể ảnh hưởng generation tiếp theo qua các activation/cache ở phía sau.

Khác biệt quan trọng: gate train trên h(x+y_neg), infer trên h(x). Đây là lệch đầu vào train–infer có thể ảnh hưởng khả năng tổng quát.

### 4.5. Baseline và steered

Mỗi tập lấy 200 mẫu, chạy hai chế độ, generation greedy (`do_sample=False`):

```text
200 U_dep    × baseline và steered
200 U_nondep × baseline và steered
= 400 bản ghi đầu vào, 800 lượt sinh theo mẫu
```

Với batch 32, mỗi nhóm 200 mẫu cần 7 batch, tổng cộng 28 lần gọi generate cho bốn nhóm theo code hiện tại. Con số 400 là số bản ghi được chọn; chưa có audit loại trùng nội dung giữa các file trong báo cáo này.

## 5. Các artifact đầu ra

| File | Nội dung và mục đích |
|---|---|
| `results.json` | Cấu hình args và tổng n/repAPI/depAPI cho bốn nhóm |
| `samples.json` | Từng mẫu: prompt, hàm gốc, expected, text sinh, a và cờ rep/dep |
| `gate_scores.json` | Từng mẫu: sample_index, id, function, a, rep, dep; không có text sinh |

Baseline ghi a=null, vì hook bị tắt; không có nghĩa a=0 do MLP dự đoán. Steered ghi gate ở lượt prompt đầu, không phải trung bình gate theo mọi token.

Code còn lưu `v_steer.pt` và `gate.pt`, nhưng hai file đó không nằm trong các artifact của thư mục run được kiểm tra trên GitHub. Chưa thể dùng trọng số để tái chạy hoặc kiểm tra norm vector từ dữ liệu đang có.

## 6. Kết quả định lượng

### 6.1. Tổng số do repo báo cáo

| Tập | Chế độ | N | repAPI | depAPI |
|---|---|---:|---:|---:|
| U_dep | Baseline | 200 | 18 (9%) | 36 (18%) |
| U_dep | Steered | 200 | 16 (8%) | 34 (17%) |
| U_nondep | Baseline | 200 | 7 (3,5%) | 1 (0,5%) |
| U_nondep | Steered | 200 | 7 (3,5%) | 1 (0,5%) |

Trên U_dep, dep giảm 1 điểm phần trăm nhưng rep cũng giảm 1 điểm phần trăm. Trên U_nondep, các tổng số không đổi. Cột rep và dep không loại trừ nhau.

### 6.2. Phân nhóm lại từ cờ từng mẫu

| Tập | Chế độ | Chỉ dep | Cả dep và rep | Chỉ rep | Không cả hai (mismatch) |
|---|---|---:|---:|---:|---:|
| U_dep | Baseline | 18 | 18 | 0 | 164 |
| U_dep | Steered | 18 | 16 | 0 | 166 |
| U_nondep | Baseline | 1 | 0 | 7 | 192 |
| U_nondep | Steered | 1 | 0 | 7 | 192 |

Đây là thống kê theo cờ đã lưu, chưa chấm lại bằng bộ dò sửa lỗi. “Chỉ rep” là rep=True và dep=False, chưa đảm bảo đúng tham số/chức năng.

Tổng 400 bản ghi:

- Có dep: 37→35, tương ứng 9,25%→8,75%.
- Chỉ rep: 7→7, giữ ở 1,75%.
- Mismatch: 356→358, tương ứng 89%→89,5%.

Mức giảm dep không đi kèm tăng replacement sạch.

### 6.3. Chuyển đổi từng mẫu

U_dep:

| Baseline → Steered | Số mẫu |
|---|---:|
| Mismatch → mismatch | 164 |
| Cả hai → cả hai | 16 |
| Chỉ dep → chỉ dep | 17 |
| Cả hai → chỉ dep | 1 |
| Cả hai → mismatch | 1 |
| Chỉ dep → mismatch | 1 |

Không có mẫu U_dep chuyển sang chỉ rep theo cờ hiện tại.

U_nondep: 191 mismatch giữ nguyên nhóm; 6 chỉ rep giữ nguyên nhóm; 1 chỉ dep giữ nguyên nhóm; 1 mismatch→rep và 1 rep→mismatch. Không đổi nhóm không có nghĩa text sinh giống nhau.

### 6.4. Gate gần như luôn mở

| Thống kê a | U_dep | U_nondep |
|---|---:|---:|
| Mean | 0.999899 | 0.999615 |
| Min | 0.998538 | 0.989342 |
| Median | 0.999986 | 0.999926 |
| Số a > 0.99 | 200/200 | 199/200 |
| Số a > 0.999 | 199/200 | 185/200 |

Với t=1, delta gần bằng v_steer cho hầu hết mẫu. MLP chưa thể hiện vai trò điều chỉnh có chọn lọc. Không thể diễn giải a≈1 là “model dự đoán gần chắc chắn sẽ sinh deprecated API”.

## 7. Kiểm tra chất lượng phép đo

Repo đếm API bằng regex trên toàn bộ văn bản sinh. Regex replacement chấp nhận tên có cùng thành phần cuối; regex không loại comment/string. Vì vậy có hai vấn đề đã được quan sát:

1. Mẫu U_dep index 99 sinh `torch.qr(A)`, expected là `torch.linalg.qr`, nhưng bị đánh dấu cả rep và dep. Chỉ dùng hậu tố `qr` không phân biệt được hai API.
2. Mẫu U_dep index 74 có lời gọi nằm trong comment, ví dụ `# return F.interpolate(...)`, vẫn được tính vào số liệu.

Do đó repAPI có thể là false positive. Số liệu chưa chứng minh code được thực thi, đúng chức năng hoặc thay API đúng. Cần chấm lại `samples.json` bằng nhận diện lời gọi và alias chính xác, bỏ comment/string, rồi giữ riêng nhóm cả hai.

Mẫu index 166 chuyển từ output có `torch.symeig(...)` sang các biểu thức `torch.max`/`torch.min`, không sang expected `torch.linalg.eigh`. Đây là ví dụ định tính cho việc mất API cũ không đồng nghĩa sửa đúng thao tác.

## 8. Vì sao run chưa đạt mục tiêu?

### 8.1. Mục tiêu gate chưa khớp vai trò mong muốn

Cosine loss chỉ khuyến khích alignment giữa h_neg đã dịch chuyển và h_pos. Không có thành phần buộc gate tắt trên mẫu vốn sinh rep đúng. Nếu tăng a thường giảm loss, gate tiến tới 1 là hành vi được objective cho phép. Gate saturation đã được quan sát; nguyên nhân tối ưu cụ thể cần loss/gradient để xác nhận thêm.

### 8.2. Gate học và sử dụng trên hai loại hidden state khác nhau

Training thấy completion deprecated; inference chưa thấy completion. Lệch phân phối này được xác nhận từ `gate_train_input=neg` và code. Chưa có run đối chứng `probe` trong các artifact đang phân tích để lượng hóa tác động.

### 8.3. Alignment hidden không bảo đảm sinh đúng chuỗi API

Các block phía sau và output head quyết định token sinh. Tăng cosine ở block 12 không bảo đảm xác suất chuỗi replacement tăng, càng không bảo đảm đúng tham số. Loss này là proxy, không trực tiếp đo mục tiêu cuối.

### 8.4. Vector và vị trí áp dụng có thể quá tổng quát

Vector chung trộn nhiều phép thay API; các hướng riêng có thể triệt tiêu hoặc phản ánh cú pháp/ngữ cảnh. Vector học ở cuối completion nhưng được áp lên mọi vị trí prompt. Đây là các giả thuyết hợp lý từ thiết kế, chưa phải nguyên nhân đã được chứng minh bằng ablation.

### 8.5. Không đủ dữ liệu để đổ lỗi riêng cho t=1

Cường độ thực phụ thuộc a, norm vector và norm hidden. Không có norm vector/hidden trong ba JSON, không có sweep hoặc đối chứng t khác. Do đó không thể kết luận t=1 quá lớn, hay giảm t chắc chắn làm phương pháp tốt hơn.

### 8.6. Baseline và protocol cần được kiểm tra

Mismatch rất cao ngay ở baseline. Có thể liên quan prompt, vị trí completion, giới hạn 48 token, truncation hoặc bộ dò; chưa thể tách nguyên nhân. Các mẫu đầu mỗi file cũng có thể không đại diện toàn tập. Run chỉ có một seed, không có functional tests hoặc uncertainty analysis.

## 9. Kết luận: phương pháp có work không?

**Về cơ chế phần mềm:** artifact cho thấy baseline và steered đã được tạo, gate có output và có thay đổi ở một số kết quả. Tuy nhiên, không có đủ log/trọng số để kiểm toán toàn bộ quá trình huấn luyện.

**Về mục tiêu sửa deprecated thành replacement:** trong run này, chưa có bằng chứng hiệu quả. Trên U_dep không có chuyển đổi sang replacement sạch theo cờ đã lưu; dep giảm chủ yếu thành mismatch. Trên U_nondep một cải thiện bị bù bởi một suy giảm.

**Về chọn lọc can thiệp:** gate gần như luôn mở, nên chưa thể hiện được chức năng nhận biết mẫu cần steering.

**Về kết luận nghiên cứu:** không nên tuyên bố phương pháp đã thành công; cũng không nên kết luận mọi biến thể của phương pháp đều không khả thi. Cách phát biểu phù hợp là:

> Với cấu hình L12, t=1 và gate học trên h_neg, thí nghiệm 200 mẫu mỗi nhóm chưa cho thấy lợi ích chuyển deprecated API sang replacement API. Gate bão hòa và hạn chế của phép đo khiến kết quả cần được kiểm chứng lại trước khi suy rộng.

## 10. Các bước tiếp theo có giá trị nhất

1. Chấm lại output đã có trước: sửa alias/hậu tố, loại comment/string; báo cáo bốn nhóm và chuyển đổi từng mẫu.
2. Đối chứng gate với a=1 trên cùng mẫu, vector, layer và t để đo gate có thêm lợi ích hay không.
3. Đối chứng `neg` với `probe`, không đồng thời đổi prompt/layer/t.
4. Nếu mục tiêu là classifier nguy cơ, tạo nhãn từ baseline generation trên tập training riêng, dùng validation để đánh giá precision/recall; không lấy nhãn test để học gate.
5. Kiểm tra norm can thiệp, các vị trí hook và vector theo nhóm API, lựa chọn trên validation.
6. Lưu commit chạy, revision model/data, số mẫu thực tế, loss từng epoch, trọng số gate/vector và thời gian. Sau khi chốt thiết lập, đánh giá test độc lập kèm chức năng nếu có thể.

## 11. Nguồn cố định để đối chiếu

- [Code algo.py](https://github.com/dinhngoctuyen4125/MLLMEraser/blob/b0ad8632ed3dc71d95234b204301964e2217b113/algo.py)
- [results.json](https://github.com/dinhngoctuyen4125/MLLMEraser/blob/b0ad8632ed3dc71d95234b204301964e2217b113/results/L12_t1.0_neg/results.json)
- [gate_scores.json](https://github.com/dinhngoctuyen4125/MLLMEraser/blob/b0ad8632ed3dc71d95234b204301964e2217b113/results/L12_t1.0_neg/gate_scores.json)
- [samples.json](https://github.com/dinhngoctuyen4125/MLLMEraser/blob/b0ad8632ed3dc71d95234b204301964e2217b113/results/L12_t1.0_neg/samples.json)

Các số đếm nhóm, chuyển đổi và thống kê gate được tính lại từ artifact. Báo cáo không chạy lại model, không sửa code của repo bạn và không sử dụng kết quả của project khác.

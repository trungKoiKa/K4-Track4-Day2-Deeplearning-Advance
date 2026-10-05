# Tiếp tục huấn luyện trên CPU Windows

`run_local_cpu.py` chạy 6 lượt riêng: T00 seed 0/1/2 (CE), F01 seed 0/1/2
(label smoothing 0,1), ConvNeXt-Tiny `convnext_tiny.fb_in1k`, mỗi lượt 10 epoch.
Batch 16, accumulation 4, FP32, 8 luồng CPU, 2 worker đọc ảnh. Mọi lượt dùng
`lab_solution.train.run(Config(...))`. Bộ code gốc, `starter/`, `eval.py` và
11 lượt sàng lọc trên Colab giữ nguyên.

Dữ liệu: `D:\Lab_VinUni\Lab_Phase2\Lab_2\data\{images,labels}`.
Kết quả và checkpoint nằm ngoài dự án tại:
`C:\Users\Hoang Anh\.codex\visualizations\2026\10\03\01a10133-0b7a-7873-a4da-69e8e881f33b\cpu_study`.

Mở PowerShell tại thư mục dự án, chạy `./Start-LocalCPU.ps1` để tiếp tục khi
worker đã dừng. Script kiểm tra PID để tránh mở hai lượt cùng ghi checkpoint.
Manifest `cpu_training_plan.json` khóa code, driver, môi trường, CSV và cấu hình.
Không sửa code hoặc đổi thư viện khi đang chạy; resume từ `last.pt` của cùng lượt.
Khi ngắt giữa epoch, lần tiếp tục chạy lại epoch chưa lưu, giữ nguyên RNG/lịch LR
của checkpoint cuối. Không tự giảm epoch hoặc đổi seed.

Đọc `cpu_status.json` và `training_stdout.log` để theo dõi. Mỗi epoch hoàn tất có
`runs/<exp>/seed<n>/history.csv`, `last.pt`, `best.pt` và curve riêng.
Máy cần tiếp tục bật và không sleep để tiến trình chạy; nếu máy tắt, dùng script
trên sau khi bật lại. Không cần giữ Colab mở.

Khi một lượt đã có `result.json`, chạy `python sync_cpu_outputs.py` từ thư mục
dự án. Script chỉ đồng bộ config, log, metric, validation logit và curve của lượt
đã hoàn tất vào `artifacts/local_cpu/`; nó loại checkpoint, cache test và dataset.
Sau đó commit/push các file nhẹ này lên GitHub.

Chế độ này chỉ hoàn tất huấn luyện. Sau 6 lượt, trạng thái vẫn là
`ready_to_submit: false`: cần hoàn tất so sánh inference trên validation, khóa
pipeline/hiệu chuẩn, chạy test có cache, đo latency trên cùng CPU, đánh giá bằng
`eval.py`, xuất và kiểm tra sáu sản phẩm. Không gộp latency CPU với T4 để xếp hạng.
Driver không mở test và không tự nộp bài.

## Chuyển một lượt đang dở sang T4

Chỉ chuyển **sau khi một epoch đã lưu checkpoint**. Dùng
`prepare_gpu_migration.py` để tạo ZIP ngoài Git; nó lưu cả metadata CPU gốc,
SHA256 checkpoint và metadata CUDA mới. Bản tiếp tục giữ `amp=False` để khôi
phục scaler/optimizer/scheduler/RNG an toàn; các lượt T4 mới có thể dùng AMP
theo cấu hình chung. Không sửa trực tiếp `config.json` hay `metadata.json` của
CPU study và không upload ZIP này lên GitHub.

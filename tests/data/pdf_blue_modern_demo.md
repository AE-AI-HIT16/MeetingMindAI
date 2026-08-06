# Báo cáo cuộc họp

## Tổng quan

Cuộc họp tập trung đánh giá **tiến độ dự án MeetingMind AI**, chất lượng bản xuất tài liệu và kế hoạch hoàn thiện trải nghiệm người dùng. Các nhóm thống nhất ưu tiên độ ổn định, khả năng bảo trì và tính nhất quán giữa DOCX và PDF.

## Mục tiêu và kết quả

- Hoàn thiện luồng xuất DOCX/PDF theo template trong tháng này.
- Giảm thời gian tạo báo cáo xuống dưới 10 giây.
- Đảm bảo nội dung tiếng Việt, bảng và danh sách hiển thị chính xác.
- Chuẩn hóa registry để bổ sung template mới mà không sửa renderer.

## Phát hiện và khuyến nghị

- Nội dung từ LLM đã có cấu trúc heading rõ ràng và phù hợp để dựng section động.
- Template cần tách biệt hoàn toàn với logic xử lý Markdown.
- Nên dùng cùng hệ màu và font cho toàn bộ bộ nhận diện.
- Cần duy trì kiểm thử với báo cáo ngắn, dài, bảng và danh sách lồng nhau.
- Tài nguyên bên ngoài phải tiếp tục bị chặn khi render PDF.

## Kế hoạch triển khai

1. Hoàn thiện template Blue Modern và kiểm tra trên khổ A4.
2. Kết nối lựa chọn template vào API export.
3. Bổ sung kiểm thử hồi quy cho font, màu và section động.
4. Đánh giá kết quả với dữ liệu cuộc họp thực tế trước khi phát hành.

## Khách hàng mục tiêu

| Nhóm người dùng | Nhu cầu chính | Giá trị mang lại |
|---|---|---|
| Quản lý dự án | Theo dõi quyết định và tiến độ | Báo cáo điều hành rõ ràng |
| Nhóm kỹ thuật | Nắm action item và phụ thuộc | Phân công có cấu trúc |
| Khách hàng | Nhận biên bản chuyên nghiệp | Tài liệu nhất quán, dễ đọc |

## Quyết định và phân công

| Thành viên | Công việc | Trạng thái |
|---|---|---|
| Anh Tú | Hoàn thiện DOCX/PDF exporter | Đang thực hiện |
| Nam | Kiểm tra giao diện chọn template | Chờ tích hợp |
| Minh | Kết nối API export | Đang chuẩn bị |
| Nhóm QA | Kiểm thử báo cáo thực tế | Chưa bắt đầu |

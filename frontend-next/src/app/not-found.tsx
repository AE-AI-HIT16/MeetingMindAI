import { ErrorPage } from "@/components/ErrorPage";

export default function NotFound() {
  return (
    <ErrorPage
      title="Không tìm thấy trang"
      message="Có thể tài liệu đã bị xóa hoặc đường dẫn không còn đúng."
    />
  );
}

# aulys

## Description
Framework này cung cấp các công cụ để phân tích dữ liệu, bao gồm các chức năng tổng hợp thông tin, trực quan hóa dữ liệu và phân tích thống kê trên các cột trong DataFrame. Các chức năng chính của dự án giúp bạn hiểu rõ hơn về dữ liệu của mình và thực hiện phân tích dữ liệu một cách hiệu quả.

### Các Chức Năng Chính:
1. **Tổng hợp các đặc trưng thống kê** cho các cột dạng số và phân loại.

2. **Tạo bảng tổng quan** về dữ liệu, bao gồm các thông tin về các giá trị trùng lặp, thiếu, và các đặc trưng khác.

3. **Vẽ biểu đồ đơn biến (univariate)** cho các cột dữ liệu.

4. **Vẽ biểu đồ phân tích đa biến (multivariate)** giữa một cột chính và các cột còn lại.

## Usage

### Installation

Install latest from the GitHub [repository][repo]:

```sh
$ pip install git+https://github.com/nguyenbaanh7779/aulys.git
```

[repo]: https://github.com/nguyenbaanh7779/aulys

## How to use

### Tổng Hợp Đặc Trưng Thống Kê

Hàm `stast_fea(df, idx_col, cat_cols, num_cols)` thực hiện việc tính toán các thống kê cho các cột số và phân loại trong DataFrame. Các thống kê này giúp bạn hiểu rõ hơn về dữ liệu của mình.

#### Thống Kê Cho Cột Số:

- **Trung bình (mean)**: Tính giá trị trung bình của các phần tử trong cột.

- **Giá trị lớn nhất (max)**: Tìm giá trị lớn nhất trong cột.

- **Giá trị nhỏ nhất (min)**: Tìm giá trị nhỏ nhất trong cột.

- **Tổng (sum)**: Tính tổng các giá trị trong cột.

- **Trung vị (median)**: Tính giá trị trung vị của các phần tử trong cột.

- **Hiệu số giữa giá trị lớn nhất và nhỏ nhất (diff)**: Tính hiệu số giữa giá trị lớn nhất và giá trị nhỏ nhất trong cột.

#### Thống Kê Cho Cột Phân Loại:

- **Số lượng giá trị khác nhau (nunique)**: Đếm số lượng các giá trị duy nhất trong cột.

- **Giá trị xuất hiện nhiều nhất (mode)**: Tìm giá trị xuất hiện nhiều nhất trong cột.

- **Tần suất xuất hiện của từng giá trị trong cột phân loại**: Hiển thị số lần xuất hiện của từng giá trị trong cột phân loại.

#### Tham Số Đầu Vào:
- **df**: DataFrame chứa dữ liệu cần tính toán thống kê.

- **idx_col**: Cột chỉ mục (ID nhóm) để phân nhóm các giá trị.

- **cat_cols**: Danh sách các cột phân loại (categorical) cần tính toán thống kê.

- **num_cols**: Danh sách các cột số (numerical) cần tính toán thống kê.

#### Ví Dụ Sử Dụng

Giả sử bạn có một DataFrame `df` và muốn tính toán các thống kê cho các cột phân loại và số, với cột `idx_col` là chỉ mục:

```python
from aulys import aulys

aulys.stast_fea(
    df, 
    idx_col='idx_col', c
    at_cols=['category1', 'category2'], 
    num_cols=['numerical1', 'numerical2']
)
```

### Thống Kê Tổng Quan Về Dữ Liệu

Hàm `overview_table(df)` giúp bạn tạo ra một bảng tổng quan cho các cột trong DataFrame, cung cấp các thông tin cơ bản về dữ liệu của bạn.

#### Các Thông Tin Tổng Quan:

Bảng tổng quan sẽ bao gồm các thông tin sau cho mỗi cột trong DataFrame:

- **Số lượng giá trị trùng lặp (Count_duplicate)**: Số lượng các giá trị bị trùng lặp trong cột.

- **Số lượng giá trị bị thiếu (Count_missing)**: Số lượng các giá trị bị thiếu (NaN) trong cột.

- **Tỷ lệ phần trăm giá trị bị thiếu (Percent_missing)**: Tỷ lệ phần trăm các giá trị bị thiếu trong cột.

- **Số lượng giá trị duy nhất (Count_distinct)**: Số lượng các giá trị khác nhau trong cột.

- **Số lượng giá trị bằng 0 (Count_zero)**: Số lượng giá trị bằng 0 trong cột (chỉ áp dụng cho các cột có kiểu số).

- **Tỷ lệ phần trăm giá trị bằng 0 (Percent_zero)**: Tỷ lệ phần trăm giá trị bằng 0 trong cột (chỉ áp dụng cho các cột có kiểu số).

#### Tham Số Đầu Vào:
- **df**: DataFrame chứa dữ liệu cần tính toán thống kê.

#### Ví Dụ Sử Dụng

Giả sử bạn có một DataFrame `df` và muốn tạo bảng tổng quan về dữ liệu:

```python
from aulys import aulys

aulys.overview_table(df=df_acc)
```

### Vẽ Biểu Đồ Đơn Biến

Hàm `plot_univariate(df)` vẽ biểu đồ đơn biến cho tất cả các cột trong DataFrame:

- **Các cột phân loại (chuỗi)** sẽ được vẽ biểu đồ tần suất.
- **Các cột số** sẽ được chia thành các bin và vẽ biểu đồ phân phối.

#### Tham Số Đầu Vào:
- **df**: DataFrame chứa dữ liệu cần tính toán thống kê.
- **bin_cols**: Từ điển chứa các giá trị phân bin cho các cột số.

#### Cấu Hình Bin

Cấu hình bin được tạo dưới dạng json gồm tên cột và các mốc chia bin với cấu trúc như sau:

```json
{
    "tên_cột_1": [x1, x2, x3, ...],
    "tên_cột_2": [x1, x2, x3, ...]
}
```

#### Ví dụ sử dụng

Nếu muốn vẽ biểu đồ đơn biến của một dataframe `df` thì ta thực hiện như sau:
```python
from aulys import aulys

bin_cols = {
    "age": [20, 30, 40, 50, 60],
    "income": [1000, 2000, 3000, 4000, 5000]
}

aulys.plot_univariate(df=df, bin_cols=bin_cols)
```

### Vẽ Biểu Đồ Phân Tích Đa Biến

Hàm `plot_multivariate(df, column)` vẽ biểu đồ phân tích đa biến giữa một cột chính (`column`) và tất cả các cột còn lại trong DataFrame.

#### Mô Tả

- Hàm sẽ phân tích mối quan hệ giữa **cột chính** (`column`) và các cột còn lại trong DataFrame.

- Tùy thuộc vào kiểu dữ liệu của các cột còn lại (cột số hoặc cột phân loại), các cột này sẽ được xử lý và chuyển đổi phù hợp để hiển thị biểu đồ **crosstab** (ma trận chéo).

#### Cách Hoạt Động:

- Nếu các cột còn lại là **cột phân loại (chuỗi)**, hàm sẽ chuyển đổi các giá trị trong cột thành các nhóm và vẽ biểu đồ crosstab.

- Nếu các cột còn lại là **cột số** (numerical), hàm sẽ phân chia các giá trị thành các bin hoặc hiển thị theo các nhóm để thể hiện mối quan hệ với cột chính.

#### Tham Số Đầu Vào:

- **df**: DataFrame chứa dữ liệu cần tính toán thống kê.

- **column**: Cột chính cần phân tích.

- **bin_cols**: DaDictionary chứa thông tin bin cho các cột số.

#### Ví Dụ Sử Dụng

Giả sử bạn muốn phân tích mối quan hệ giữa cột `column` (một cột số) và các cột còn lại trong DataFrame, bạn có thể sử dụng hàm như sau:

```python
from aulys import aulys

bin_cols = {
    "age": [20, 30, 40, 50, 60],
    "income": [1000, 2000, 3000, 4000, 5000]
}
aulys.plot_multivariate(df=df_acc, column='column')
```
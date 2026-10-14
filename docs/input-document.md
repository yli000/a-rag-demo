# PDF 输入说明

当前配置使用上汽大众《全新一代 Polo 系列车型使用维护说明书》，出版日期为 2021 年 1 月，版本编号 67D00401121，文档编号 KBA67D00421。PDF 共 212 页。

- [上汽大众官方使用维护说明书入口](https://techcare.svw-volkswagen.com/khgg/sywh.html)
- [官方 Polo 说明书链接](https://techcare.svw-volkswagen.com/pdf-1/sywh/Polo%E4%BD%BF%E7%94%A8%E7%BB%B4%E6%8A%A4%E8%AF%B4%E6%98%8E%E4%B9%A6.pdf)
- 本地输入文件的 SHA-256：`5b443c6b88ed0d9ba719154a817b80383e63d656480f74eeda240d5ae0e83882`

原下载地址未保存，上述链接是目前找到的官方获取入口，未确认线上文件与本地文件逐字节一致。PDF 原文不包含在仓库中，请下载后在本地配置输入路径。

图片提取保留 Polo 版式的适配：在图片下方 50pt 范围内搜索图题，按纵坐标优先匹配最近文字，并要求图题以“图＋数字”开头，以减少正文图号引用的干扰。

替换 PDF 时，需要在 `src/parser/pdf_parse.py` 与 `src/parser/image_handler.py` 中调整页码范围、裁切和图题关联规则。网页或页面截图不是运行所需资源。

# 鏈碼正規化

<img width="1054" height="386" alt="圖片" src="https://github.com/user-attachments/assets/6f11499f-dcb3-4e5d-92c7-e0a4203b28e6" />

- 由第一行 -> 3 3 0 0 1 1 2 2
- 把第一個移到最後面 -> 3 0 0 1 1 2 2 3

這樣排完後轉整數找最小的 -> ans

```txt
0 0 1 1 2 2 3 3 -> 112233
```
<img width="1109" height="673" alt="圖片" src="https://github.com/user-attachments/assets/0f5f39c6-3f02-4eed-8bd9-4becaa15e91d" />

形狀數(旋轉)

- c 原本沒改過
- c1 位移 (旋轉) 過

- (c - c1) % 4 => 要旋轉的角度

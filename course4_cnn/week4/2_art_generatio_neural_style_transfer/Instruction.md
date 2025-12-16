# Neural Style Transfer

1. Load một pretrained VGG model
```python
img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False,
                                    input_shape=(img_size, img_size, 3),
                                    weights='imagenet')
```
    
2. Load content image $C$ và style image $S$
```python
content_image = np.array(Image.open("content_image.jpg").resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

style_image =  np.array(Image.open("style_image.jpg").resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
```
    
3. Khởi tạo generated image $G$ từ content image cộng thêm nhiễu. Việc này sẽ giúp content of generated image nhanh chóng match với content of content image.
```python
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), minval=0, maxval=0.8)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
```
    
4. Chọn một middile layer để tính activation output của một input image, sẽ biểu diễn content của image đó. Tính activation output $a^{(C)}$ của content image và $a^{(S)}$ của style image.
```python
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)
```
    
5. Trong một vòng lặp training

    a. Tính activation output $a^{(G)}$ của generated image.
    ```python
    a_G = vgg_model_outputs(generated_image)
    ```
        
    b. Tính content cost giữa generated image và content image
    ```python
    J_content = compute_content_cost(a_C, a_G)
    ```
    Giả sử các images có size $n_H\times n_W \times n_C$. Content cost là: 
    $$
    J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2
    $$
    
    $a^{(C)}$ và $a^{(G)}$ là các 3D tensor. Để tính toán thuận tiện ta chuyển chúng thành các 2D matrices.
    ![](images/NST_LOSS.png)
        
    c. Tính style cost giữa generated image và style image
    ```python
    J_style = compute_style_cost(a_S, a_G)
    ```
    Style của một image đc tính bằng Gram matrix. Cho một tập vectors $\left(v_1, \dots, v_n\right)$, nó đc tính là $G_{ij} = v_i^T v_j$. $G_{ij}$ so sánh độ tương đồng giữa $v_i$ và $v_j$. Nếu chúng giống nhau sẽ cho ra giá trị $G_{ij}$ lớn.
    Ta tính Gram matrix bằng cách nhân unrolled filter matrix với transpose của nó:
    $$
    \mathbf{G}_{gram} = \mathbf{A}_{unrolled} \mathbf{A}_{unrolled}^T.
    $$
    ![](images/NST_GM.png)
    KQ là một ma trận $n_C \times n_C$. $G_{(gram)i,j}$ đo độ tương đồng giữa các hai activations của filter $i$ và $j$. Các diagonal elements $G_{(gram)ii}$ đo độ "active" của một filter $i$. VD, giả sử filter $i$ đang phát hiện vertical textures trong image, thì $G_{(gram)ii}$ sẽ đo độ phổ biến của vertical textures trong toàn bộ image. $G_{(gram)ii}$ lớn nghĩa là image có nhiều vertical texture.
    
    Bằng cách đo độ phổ biến của các loại features khác nhau bằng $G_{(gram)ii}$, cũng như đo việc các features khác nhau xuất hiện cùng nhau bằng $G_{(gram)ij}$, ma trận $G_{gram}$ sẽ đo style của một image.
    
    Style lost cho layer $l$ là:
    $$
    J^{[l]}_{style}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2
    $$
    Sẽ có KQ tốt hơn nếu merge nhiều style cost từ vài layers khác nhau. Mỗi layer sẽ đc gán một weight $\lambda^{[l]}$, sao cho $\sum \lambda^{[l]}=1$. Combined style cost là:
    $$
    J_{style}(S,G) = \sum_{l} \lambda^{[l]} J^{[l]}_{style}(S,G)
    $$
        
    d. Tính total cost:
    ```python
    J = total_cost(J_content, J_style, alpha, beta)
    ```
    
    Công thức tính total cost:
    
    $$
    J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)
    $$
        

### Notes
* Content cost function đc tính chỉ dùng một hidden layer’s activation.
* Style cost function cho một layer đc tính bằng Gram matrix của layer’s activation đó. Overall cost function đc tính bằng một số hidden layers.
* Có thể thay đổi các hidden layers đc dùng để cho ra các KQ khác nhau.
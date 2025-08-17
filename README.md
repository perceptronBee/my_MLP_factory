# 🧠 my_MLP_factory

Basit bir **Python Multi-Layer Perceptron (MLP) sınıfı**.  
Kendi küçük yapay sinir ağı deneylerini yapmak için tasarlandı. 🎯

---

## ✨ Özellikler

- Çok katmanlı perceptron desteği (gizli ve çıkış katmanları)
- Sigmoid aktivasyon fonksiyonu
- Basit feedforward ve backpropagation
- Kendi veri setinle veya **XOR** gibi örneklerle çalışabilir

---

## ⚡ Kurulum

1. Python 3.x yüklü olmalı 🐍  
2. Dosyayı indir veya klonla  
3. Ekstra paket yok, tamamen standart Python ile çalışıyor ✔️

---

## 🛠️ Kullanım Örneği

```python
from my_MLP_factory import MLP

# XOR veri seti
data = [[0,0],[0,1],[1,0],[1,1]]
labels = [[0],[1],[1],[0]]

# MLP oluşturuluyor
# [2,2,1] -> Katman yapısı:
# 2 nöronlu input layer (girdi sayısı=2)
# 2 nöronlu gizli layer
# 1 nöronlu output layer
# 2 -> input katmanındaki her bir nöron için giriş sayısı
mlp = MLP([2,2,1], 2)

epochs = 50000  # kaç defa tüm veri seti üzerinden geçileceği
lr = 0.5        # öğrenme hızı

for epoch in range(epochs):
    loss = mlp.train_epoch(data, labels, lr)
    if epoch % 10000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

print("\nXOR Test Sonuçları:")
for x, y in zip(data, labels):
    out = mlp.feedforward(x, y)
    print(f"{x} -> {out}")

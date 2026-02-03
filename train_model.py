import torch
import torch.nn as nn
import torch.optim as optim
from model import PerformancePredictor
from preprocess import X_train, y_train, X_test, y_test

model = PerformancePredictor()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")

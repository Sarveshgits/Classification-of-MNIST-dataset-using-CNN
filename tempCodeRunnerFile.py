for i in range(epochs):
#     y_pred = model.forward(X_train)

#     loss=criterion(y_pred, y_train)
#     losses.append(loss.detach().numpy())

#     if i % 10==0:
#         print(f'Epoch: {i} and loss: {loss}')

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


# with torch.no_grad():
#     y_eval=model.forward(X_test)
#     loss=criterion(y_eval,y_test)

# print(loss)

# correct=0
# with torch.no_grad():
#     for i, data in enumerate(X_test):
#         y_val=model.forward(data)

#         print(f'{i+1}.) {str(y_val)} \t {y_test[i]} \t{y_val.argmax().item()}')

#         if y_val.argmax().item()==y_test[i]:
#             correct+=1
# print(f'We got {correct} correct!')
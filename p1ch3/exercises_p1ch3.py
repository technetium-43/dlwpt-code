import torch


def random_tests():
    data_gpu = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float, device='cuda')
    data_gpu_t = torch.transpose(data_gpu, 0, 1)

    q = torch.mm(data_gpu, data_gpu_t)
    print(q)


def exercise_3_14_1():
    # Exercise 3.14 part 1
    a = torch.tensor(range(9))
    print(a)
    print("a size:", a.size())  # size =  [9]
    print("a offset:", a.storage_offset())  # offset = 0
    print("a stride:", a.stride())  # stride = (1,)

    # Create a new tensor and evaluate the .view() method and storage
    b = a.view(3, 3)
    print(b)
    print("b size:", b.size())  # size = [3, 3]
    print("b offset:", b.storage_offset())  # offset = 0
    print("b stride:", b.stride())  # stride = (3, 1)

    # Check the storage is still the same.
    # print("a storage", a.storage())
    print("b storage after view", b.storage())
    print("Same storage: ", id(a.storage()) == id(b.storage()))

    # We could update the storage to be contiguous
    b = b.contiguous()
    print("Same storage: ", id(a.storage()) == id(b.storage()))

    # Create a new tensor & evaluate the size, offset, stride
    c = b[1:, 1:]
    print(c)
    print("c size:", c.size())  # size = [2, 2]
    print("c offset:", c.storage_offset())  # offset = 4
    print("c stride:", c.stride())  # stride = (2, 1)


def exercise_3_14_2():
    # Exercise 3.14 part 2

    # Calculate the square root element-wise
    # https://pytorch.org/docs/stable/generated/torch.sqrt.html?highlight=square%20root
    a = torch.tensor(range(9), dtype=torch.float)
    a_sqrt = a.sqrt()
    print("a:", a)
    print("a_sqrt:", a_sqrt)

    # Perform the operation in place
    a.sqrt_()
    print("a updated in place:", a)


# Check the results to 3.14 Exercises
exercise_3_14_1()
exercise_3_14_2()

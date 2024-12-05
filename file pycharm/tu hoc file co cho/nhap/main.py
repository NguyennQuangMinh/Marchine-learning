list1=[1,2,3]
list2=[4,5]
list3=list1+[7,8,9]+list2
print('chuỗi 3',list3)
# Sắp xếp theo mặc định tăng dần
list3.sort()
print('Chuỗi sau khi sắp xếp tăng dần : ',list3)
# Sắp xếp theo thữ tự giảm dần
list3.sort(reverse=True)
print('chuoi sau khi sap xep giam dan',list3)
b={'một':1,'two':2, 'three':3}
print('Dictionary b là : ',b)
x=input("Nhập số x = ");
y=input("Nhập số y = ");
print("Tổng của ",x,"Và",y,"là",int(x)+int(y));
z=input("nhập số kiểu float = ");
z=float(z);
print(z);
print(type(z));
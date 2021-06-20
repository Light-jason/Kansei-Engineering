import tkinter as tk
from PIL import Image, ImageTk
import pygal
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


window=tk.Tk()

width,height=1000,600
screenwidth = window.winfo_screenwidth()
screenheight = window.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
window.geometry(alignstr)
window.resizable(width=False, height=False)
window.title('用户登录')

canvas=tk.Canvas(master=window,width=1000,height=600,bg='DarkSlateblue',highlightthickness=0)
canvas.pack()
img = Image.open('背景图.jpg')  # 打开图片
photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
p=tk.Label(canvas,image=photo,highlightthickness=0,borderwidth=0)
p.place(x=30,y=50)
canvas.create_rectangle(600,180,602,420,fill='lightgrey',outline='lightgrey')
canvas.create_text(680,200,text='Login',font=('Time new roman',18,'bold'),fill='white')
userlogin=tk.Entry(master=canvas,width=30,borderwidth=0)
password=tk.Entry(master=canvas,width=30,borderwidth=0)
userlogin.place(x=650,y=240,height=30)
password.place(x=650,y=290,height=30)
login_button=tk.Button(master=canvas,width=14,text='登录',font=('宋体',18,'bold'),
                       fg='Darkslateblue',bg='SlateBlue',highlightthickness=0,borderwidth=0)
login_button.place(x=650,y=380,width=210,height=30)
canvas1=tk.Canvas(master=canvas,width=1000,height=80,bg='white',highlightthickness=0)
canvas1.place(x=0,y=0)
canvas1.create_text(490,55,text='汽车感性工学应用系统',font=('黑体',18,'bold'),fill='Darkslateblue')
canvas2=tk.Canvas(master=canvas,width=1000,height=50,bg='white',highlightthickness=0)
canvas2.place(x=0,y=550)
canvas2.create_text(490,12,text='工学二号馆615有限公司 技术支持',font=('宋体',10,),fill='Dimgrey')
canvas2.create_text(490,30,text='电话：135****5155 邮箱：Jamsonlight@163.com',font=('宋体',10,),fill='Dimgrey')

# canvas.create_text(80,70,text='账号：',font=('微软雅黑',10,'bold'),fill='black')
# canvas.create_text(80,100,text='密码：',font=('微软雅黑',10,'bold'),fill='black')
# canvas.create_rectangle(200,100,800,500,fill='white',outline='grey',dash=(40,20))

# main_view=tk.Canvas(master=canvas,width=600,height=400,bg='white',
#                     borderwidth=0,selectborderwidth=0,insertborderwidth=0,highlightthickness=0)
# main_view.config(highlightthickness=0)
# main_view.place(x=200,y=100)

# login_canvas=tk.Canvas(master=canvas,width=350,height=250,bg='white')
# login_canvas.place(x=700,y=50)


# login_canvas.create_text(160,25,text='用户登录',font=('微软雅黑',14,'bold'),fill='black')
# login_canvas.create_text(80,70,text='账号：',font=('微软雅黑',10,'bold'),fill='black')
# login_canvas.create_text(80,100,text='密码：',font=('微软雅黑',10,'bold'),fill='black')
# userlogin=tk.Entry(master=login_canvas,width=20,)
# password=tk.Entry(master=login_canvas,width=20,)
# userlogin.place(x=100,y=60)
# password.place(x=100,y=90)

# c = tk.Checkbutton(master=login_canvas,text='设计师',font=('微软雅黑',10,'bold'),fg='DimGrey')
# c.place(x=60,y=120)

# c1 = tk.Checkbutton(master=login_canvas,text='消费者/经销商',font=('微软雅黑',10,'bold'),fg='DimGrey')
# c1.place(x=130,y=120)

# login_button=tk.Button(master=login_canvas,text='登录',font=('微软雅黑',10,'bold'),fg='black',bg='LightBlue')
# login_button.place(x=115,y=150)

# log_button=tk.Button(master=login_canvas,text='注册',font=('微软雅黑',10,'bold'),fg='DimGrey')
# log_button.place(x=185,y=150)

window.mainloop()
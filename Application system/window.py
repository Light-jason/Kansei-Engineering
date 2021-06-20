import tkinter as tk
from PIL import Image, ImageTk
import pygal
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 用Matplotlib画出雷达图
def establish_leida(arrs):
    labels = ['Elegant', 'Streamlined', 'Grand', 'High-Tech', 'Fashion', 'Tough', 'Steady', 'Stylish']
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1, polar=True)  # 设置第一个坐标轴为极坐标体系

    angle = np.linspace(0, 2 * np.pi, len(arrs), endpoint=False)  # data里有几个数据，就把整圆360°分成几份
    angles = np.concatenate((angle, [angle[0]]))  # 增加第一个angle到所有angle里，以实现闭合
    data1 = np.concatenate((arrs, [arrs[0]]))

    ax1.set_thetagrids(angles * 180 / np.pi, labels)  # 设置网格标签
    ax1.plot(angles, data1, "o-", )
    # for x, y in zip(angles, data1):
    #     ax1.text(x, y, '%.2f' % y, fontdict={'fontsize': 10})

    ax1.set_theta_zero_location('NW')  # 设置极坐标0°位置
    ax1.set_rlim(0, 12)  # 设置显示的极径范围
    ax1.fill(angles, data1, facecolor='g', alpha=0.2)  # 填充颜色
    ax1.set_rlabel_position(255)  # 设置极径标签位置
    fig.set_size_inches(3, 2.4)
    # return fig
    plt.savefig('./leida.png')




window=tk.Tk()
width,height=1600,700
screenwidth = window.winfo_screenwidth()
screenheight = window.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
window.geometry(alignstr)
window.resizable(width=False, height=False)
window.title('Kansei Engineering Application System')

canvas1 = tk.Canvas(master=window,width=200, height=700, highlightthickness=0, borderwidth=0,bg='DarkSlateblue')
canvas1.place(x=0,y=0)
canvas1.create_rectangle(20,45,180,47,fill='DarkGrey',outline='DarkGrey')
canvas1.create_rectangle(20,51,180,53,fill='DarkGrey',outline='DarkGrey')

canvas1.create_text(100, 30, text='Kansei System',font = ('Times New Roman',18,'bold'),fill='white',)

user_list=tk.StringVar()
user_list.set('Kansei-Design 　　　 Vehicle-Report')
user=tk.Listbox(master=canvas1,listvariable=user_list,background='DarkSlateblue',borderwidth=0,
                highlightcolor='DarkSlateblue',selectbackground='DarkSlateblue',foreground='white',width=20,
                highlightbackground='DarkSlateblue',height=4,font=('Times New Roman',14,'bold'),)

user.place(x=35,y=90)
def application_select(self):
    select=user.get(user.curselection())
    if select=='Kansei-Design':
        #创建第二块画布：设计师功能
        canvas2=tk.Canvas(master=window,width=1400, height=700, highlightthickness=0, borderwidth=0,bg='Lavender')
        canvas2.place(x=200,y=0)

        design=canvas2.create_rectangle(20,20,550,400,fill='white',outline='white')#存放设计选择
        other=canvas2.create_rectangle(570,20,880,300,fill='white',outline='white')#存放模型建议
        model=canvas2.create_rectangle(20,420,550,680,fill='white',outline='white')#存放用户选择
        leida=canvas2.create_rectangle(570,320,880,680,fill='white',outline='white')#存放雷达图
        pipei=canvas2.create_rectangle(900,20,1380,680,fill='white',outline='white')#匹配类似车款

        # img_canvas = tk.Canvas(canvas2, bg='white', height=360, width=310)
        # img_canvas.place(x=570,y=320)

        canvas2.create_rectangle(26,30,30,50,fill='DarkSlateblue',outline='DarkSlateblue')
        canvas2.create_text(170,40,text='Design Features（Select Below）',font = ('黑体',12),fill='black',)
        canvas2.create_rectangle(30, 380, 540, 382, fill='LightGrey', outline='LightGrey')
        canvas2.create_rectangle(30, 390, 540, 392, fill='LightGrey', outline='LightGrey')

        canvas2.create_rectangle(576,30,580,50,fill='DarkSlateblue',outline='DarkSlateblue')
        canvas2.create_text(620,40,text='Advices',font = ('黑体',12),fill='black',)
        canvas2.create_rectangle(580, 280, 870, 282, fill='LightGrey', outline='LightGrey')
        canvas2.create_rectangle(580, 290, 870, 292, fill='LightGrey', outline='LightGrey')


        canvas2.create_rectangle(26,430,30,450,fill='DarkSlateblue',outline='DarkSlateblue')
        canvas2.create_text(160,440,text='User Features（Select Below）',font = ('黑体',12),fill='black',)
        canvas2.create_rectangle(30, 660, 540, 662, fill='LightGrey', outline='LightGrey')
        canvas2.create_rectangle(30, 670, 540, 672, fill='LightGrey', outline='LightGrey')

        canvas2.create_rectangle(576,330,580,350,fill='DarkSlateblue',outline='DarkSlateblue')
        canvas2.create_text(640,340,text='Images Score',font = ('黑体',12),fill='black',)
        canvas2.create_rectangle(580, 660, 870, 662, fill='LightGrey', outline='LightGrey')
        canvas2.create_rectangle(580, 670, 870, 672, fill='LightGrey', outline='LightGrey')

        canvas2.create_rectangle(906,30,910,50,fill='DarkSlateblue',outline='DarkSlateblue')
        canvas2.create_text(975,40,text='Match Vehicle',font = ('黑体',12),fill='black',)
        canvas2.create_rectangle(910, 660, 1370, 662, fill='LightGrey', outline='LightGrey')
        canvas2.create_rectangle(910, 670, 1370, 672, fill='LightGrey', outline='LightGrey')


        # 设计要素部分
        lg_canvas=tk.Canvas(master=canvas2,width=150,height=100,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        lg_canvas.place(x=30,y=80)
        lg_canvas.create_text(55, 20, text='Wheel-hub', font=('黑体', 12,'bold'), fill='DarkSlateBlue')
        lg_canvas.create_rectangle(10,30,120,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        lg=tk.StringVar()
        lg.set(("Five-spoke","Double five-spoke","Multi-spoke"))
        lungu=tk.Listbox(master=lg_canvas, listvariable=lg, background='white', borderwidth=0,
                   highlightcolor='white', selectbackground='DarkSlateBlue', foreground='black', width=20,
                   highlightbackground='white',yscrollcommand=True, height=3, font=('黑体',12,),exportselection=False)
        lungu.place(x=14,y=38)

        tx_canvas=tk.Canvas(master=canvas2,width=120,height=100,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        tx_canvas.place(x=220,y=80)
        tx_canvas.create_text(35, 20, text='Shape', font=('黑体', 12,'bold'), fill='DarkSlateBlue')
        tx_canvas.create_rectangle(10,30,120,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        tx=tk.StringVar()
        tx.set(("Compact","Normal","Slender"))
        tixing=tk.Listbox(master=tx_canvas, listvariable=tx, background='white', borderwidth=0,
                   highlightcolor='white', selectbackground='DarkSlateBlue', foreground='black', width=14,
                   highlightbackground='white',yscrollcommand=True, height=3, font=('黑体',12,),exportselection=False)
        tixing.place(x=12,y=38)

        qdd_canvas=tk.Canvas(master=canvas2,width=120,height=120,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        qdd_canvas.place(x=410,y=80)
        qdd_canvas.create_text(53, 20, text='Headlight', font=('黑体', 12,'bold'), fill='DarkSlateBlue')
        qdd_canvas.create_rectangle(10,30,120,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        qdd=tk.StringVar()
        qdd.set(("Rectangle","Willow leaf","Triangle","Irregular"))
        qddeng=tk.Listbox(master=qdd_canvas, listvariable=qdd, background='white', borderwidth=0,
                   highlightcolor='white', selectbackground='DarkSlateBlue', foreground='black', width=14,
                   highlightbackground='white',yscrollcommand=True, height=4, font=('黑体',12,),exportselection=False)
        qddeng.place(x=12,y=38)

        gs_canvas=tk.Canvas(master=canvas2,width=160,height=140,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        gs_canvas.place(x=30,y=200)
        gs_canvas.create_text(30, 20, text='grille', font=('黑体', 12,'bold'), fill='DarkSlateBlue')
        gs_canvas.create_rectangle(10,30,120,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        gs=tk.StringVar()
        gs.set(("Banner shape","Hexagon","Octagon","Irregular","Inverted trapezoid","None"))
        geshan=tk.Listbox(master=gs_canvas, listvariable=gs, background='white', borderwidth=0,
                   highlightcolor='white', selectbackground='DarkSlateBlue', foreground='black', width=20,
                   highlightbackground='white',yscrollcommand=True, height=6, font=('黑体',12,),exportselection=False)
        geshan.place(x=12,y=38)

        wd_canvas=tk.Canvas(master=canvas2,width=150,height=140,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        wd_canvas.place(x=220,y=200)
        wd_canvas.create_text(53, 20, text='Taillight', font=('黑体', 12,'bold'), fill='DarkSlateBlue')
        wd_canvas.create_rectangle(10,30,120,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        wd=tk.StringVar()
        wd.set(("Irregular","Pulling-through","Rectangle","Trapezoid"))
        weideng=tk.Listbox(master=wd_canvas, listvariable=wd, background='white', borderwidth=0,
                   highlightcolor='white', selectbackground='DarkSlateBlue', foreground='black', width=20,
                   highlightbackground='white',yscrollcommand=True, height=4, font=('黑体',12,),exportselection=False)
        weideng.place(x=12,y=38)

        car_canvas=tk.Canvas(master=canvas2,width=120,height=140,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        car_canvas.place(x=410,y=200)
        car_canvas.create_text(30, 20, text='Type', font=('黑体', 12,'bold'), fill='DarkSlateBlue')
        car_canvas.create_rectangle(10,30,120,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        car=tk.StringVar()
        car.set(("New energy","Fuel"))
        c=tk.Listbox(master=car_canvas, listvariable=car, background='white', borderwidth=0,
                   highlightcolor='white', selectbackground='DarkSlateBlue', foreground='black', width=14,
                   highlightbackground='white',yscrollcommand=True, height=4, font=('黑体',12,),exportselection=False)
        c.place(x=12,y=38)

        #用户要素部分
        city_canvas=tk.Canvas(master=canvas2,width=120,height=140,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        city_canvas.place(x=30,y=480)
        city_canvas.create_text(30, 20, text='City', font=('黑体', 12,'bold'), fill='DarkSlateBlue')
        city_canvas.create_rectangle(10,30,120,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        cy=tk.StringVar()
        cy.set(("First-tier","Second-tier","Third-tier","Fourth-tier"))
        city=tk.Listbox(master=city_canvas, listvariable=cy, background='white', borderwidth=0,
                   highlightcolor='white', selectbackground='DarkSlateBlue', foreground='black', width=14,
                   highlightbackground='white',yscrollcommand=True, height=4, font=('黑体',12,),exportselection=False)
        city.place(x=12,y=38)

        money_canvas=tk.Canvas(master=canvas2,width=150,height=140,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        money_canvas.place(x=200,y=480)
        money_canvas.create_text(75, 20, text='Purchase ability', font=('黑体', 12,'bold'), fill='DarkSlateBlue')
        money_canvas.create_rectangle(0,30,120,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        my=tk.StringVar()
        my.set(("100000","100000-200000","200000-300000","Above 300000"))
        money=tk.Listbox(master=money_canvas, listvariable=my, background='white', borderwidth=0,
                   highlightcolor='white', selectbackground='DarkSlateBlue', foreground='black', width=14,
                   highlightbackground='white',yscrollcommand=True, height=4, font=('黑体',12,),exportselection=False)
        money.place(x=0,y=38)

        goal_canvas=tk.Canvas(master=canvas2,width=150,height=160,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        goal_canvas.place(x=380,y=480)
        goal_canvas.create_text(80, 20, text='Purpose (Multi)', font=('黑体', 12,'bold'), fill='DarkSlateBlue')
        goal_canvas.create_rectangle(10,30,120,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        md=tk.StringVar()
        md.set(("Shopping", "Commute","Pick up children","Business","Appointment","Racing","Cross-country",
                "Online car hailing","Delivery","Road Trip","Fleet","Refitting","Long distance travel"))
        goal=tk.Listbox(master=goal_canvas, listvariable=md, background='white', borderwidth=0,
                   highlightcolor='white', selectbackground='DarkSlateBlue', foreground='black', width=20,
                   highlightbackground='white',yscrollcommand=True, height=7, font=('黑体',12,),
                   exportselection=False,selectmode='multiple')
        goal.place(x=12,y=38)

        #模型建议部分
        '''用户关注度排行榜'''
        rank_canvas=tk.Canvas(master=canvas2,width=240,height=140,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        rank_canvas.place(x=600,y=80)
        rank_canvas.create_text(120, 20, text='User attention ranking', font=('黑体', 11,'bold'), fill='DarkSlateBlue')
        rank_canvas.create_rectangle(10,30,240,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        rank_text = tk.Text(rank_canvas, width=50, height=5,font=('黑体', 11),  fg='black',borderwidth=0)
        top_k=['1.Willow leaf headlight\n','2.Hexagon grille \n','3.Pulling-through taillight\n','4.Five-spoke wheel-hub\n','5.Rectangle headlight\n']
        for i in range(1,len(top_k)+1):
            rank_text.insert(float(i),top_k[i-1])
        rank_text.place(x=25,y=38)

        # '''组合推荐'''
        # rank_canvas=tk.Canvas(master=canvas2,width=240,height=100,highlightthickness=0, borderwidth=0,highlightcolor='white',bg='white')
        # rank_canvas.place(x=600,y=170)
        # rank_canvas.create_text(120, 20, text='Recommendation', font=('黑体', 11,'bold'), fill='DarkSlateBlue')
        # rank_canvas.create_rectangle(10,30,240,32,fill='DarkSlateBlue',outline='DarkSlateBlue')
        # rank_text = tk.Text(rank_canvas, width=40, height=3,font=('黑体', 10),  fg='black',borderwidth=0)
        # top_k=['1.Willow leaf headlight + Five-spoke wheel-hub\n',
        #        '2.Hexagon grille + Rectangle headlight\n',
        #        '3.Pulling-through taillight + Slender shape\n']
        # for i in range(1,len(top_k)+1):
        #     rank_text.insert(float(i),top_k[i-1])
        # rank_text.place(x=25,y=38)

        #雷达图部分
        '''计算部分'''
        def cal_function():
            ry_dict={
                'Five-spoke wheel-hub':[0,0,0,0,0,0,0,0],
                'Double five-spoke wheel-hub':[-0.07,-0.07,-0.062,-0.062,-0.07,-0.074,-0.06,-0.045],
                'Multi-spoke wheel-hub': [-0.119,-0.12,-0.105,-0.106,-0.12,-0.127,-0.102,-0.077],

                'Irregular taillight':[-0.217,-0.219,-0.193,-0.195,-0.219,-0.232,-0.187,-0.14],
                'Pulling-through taillight': [0.001,0.001,0,0,0.001,0.001,0,0],
                'Rectangle taillight':[0,0,0,0,0,0,0,0],
                'Trapezoid taillight':[-0.095,-0.096,-0.084,-0.085,-0.096,-0.102,-0.082,-0.061],

                'Rectangle headlight':[-0.104,-0.105,-0.093,-0.094,-0.105,-0.112,-0.09,-0.067],
                'Triangle headlight': [-0.125,-0.126,-0.111,-0.112,-0.127,-0.134,-0.108,-0.081],
                'Willow leaf headlight': [-0.033,-0.034,-0.03,-0.03,-0.034,-0.036,-0.029,-0.022],
                'Irregular headlight':[0,0,0,0,0,0,0,0],

                'Hexagon grille':[-0.141,-0.142,-0.125,-0.126,-0.142,-0.151,-0.121,-0.091],
                'None grille':[-0.131,-0.132,-0.116,-0.117,-0.132,-0.14,-0.112,-0.084],
                'Octagon grille':[-0.166,-0.167,-0.147,-0.149,-0.168,-0.178,-0.143,-0.107],
                'Inverted trapezoid grille': [0,0,0,0,0,0,0,0],
                'Banner shape grille':[0,0,0,0,0,0,0,0],
                'Irregular grille':[-0.278,-0.28,-0.246,-0.249,-0.28,-0.297,-0.239,-0.179],

                'Compact shape':[-0.073,-0.073,-0.064,-0.065,-0.073,-0.078,-0.063,-0.047],
                'Normal shape':[-0.019,-0.019,-0.017,-0.017,-0.019,-0.02,-0.016,-0.012],
                'Slender shape':[0,0,0,0,0,0,0,0],

                'Refitting': [0.004,0.004,0.003,0.003,0.004,0.004,0.003,0.002],
                'Delivery': [-0.015,-0.015,-0.013,-0.013,-0.015,-0.016,-0.013,-0.008],
                'Racing': [-0.013,-0.013,-0.012,-0.012,-0.013,-0.014,-0.011,-0.008],
                'Long distance travel': [0.029,0.029,0.025,0.026,0.029,0.031,0.025,0.016],
                'Cross-country': [-0.023,-0.023,-0.02,-0.021,-0.023,-0.025,-0.02,-0.013],
                'Pick up children': [-0.006,-0.006,-0.005,-0.005,-0.006,-0.006,-0.005,-0.003],
                'Online car hailing': [0.046,0.046,0.04,0.041,0.046,0.049,0.039,0.026],
                'Commute': [0.031,0.031,0.027,0.028,0.031,0.033,0.027,0.018],
                'Fleet': [0.04,0.04,0.035,0.036,0.04,0.043,0.034,0.022],
                'Appointment': [0.032,0.032,0.028,0.029,0.032,0.034,0.028,0.018],
                'Trip': [0.006,0.006,0.005,0.005,0.006,0.006,0.005,0.003],
                'Business': [-0.004,-0.004,-0.003,-0.003,-0.004,-0.004,-0.003,-0.002],
                'Shopping':[0.038,0.038,0.033,0.034,0.038,0.04,0.032,0.021],

                '-100000': [0.031,0.028,0.063,0.021,0.035,0.025,0.024,0.359],
                '100000-200000': [0.03,0.026,0.06,0.02,0.033,0.024,0.023,0.343],
                '200000-300000': [0.02,0.018,0.041,0.013,0.023,0.016,0.015,0.233],
                'Above 300000': [0, 0, 0, 0, 0, 0, 0, 0],
                'Fourth-tier city': [0, 0, 0, 0, 0, 0, 0, 0],
                'First-tier city': [0.02,0.018,0.04,0.013,0.022,0.016,0.015,0.229],
                'Second-tier city': [0.015,0.013,0.03,0.01,0.017,0.012,0.011,0.173],
                'Third-tier city': [0.028,0.024,0.055,0.018,0.031,0.022,0.021,0.316],
            }
            xny_dict={
                'Five-spoke wheel-hub':[0,0,0,0,0,0,0,0],
                'Double five-spoke wheel-hub': [-0.07, -0.07, -0.062, -0.062, -0.07, -0.074, -0.06, -0.045],
                'Multi-spoke wheel-hub': [-0.119, -0.12, -0.105, -0.106, -0.12, -0.127, -0.102, -0.077],

                'Irregular taillight': [-0.217, -0.219, -0.193, -0.195, -0.219, -0.232, -0.187, -0.14],
                'Pulling-through taillight': [0.001, 0.001, 0, 0, 0.001, 0.001, 0, 0],
                'Rectangle taillight': [0, 0, 0, 0, 0, 0, 0, 0],
                'Trapezoid taillight': [-0.095, -0.096, -0.084, -0.085, -0.096, -0.102, -0.082, -0.061],


                'Rectangle headlight':[-0.104,-0.105,-0.093,-0.094,-0.105,-0.112,-0.09,-0.067],
                'Triangle headlight': [-0.125,-0.126,-0.111,-0.112,-0.127,-0.134,-0.108,-0.081],
                'Willow leaf headlight': [-0.033,-0.034,-0.03,-0.03,-0.034,-0.036,-0.029,-0.022],
                'Irregular headlight':[0,0,0,0,0,0,0,0],

                'Hexagon grille':[-0.141,-0.142,-0.125,-0.126,-0.142,-0.151,-0.121,-0.091],
                'None grille':[-0.131,-0.132,-0.116,-0.117,-0.132,-0.14,-0.112,-0.084],
                'Octagon grille':[-0.166,-0.167,-0.147,-0.149,-0.168,-0.178,-0.143,-0.107],
                'Inverted trapezoid grille': [0,0,0,0,0,0,0,0],
                'Banner shape grille':[0,0,0,0,0,0,0,0],
                'Irregular grille':[-0.278,-0.28,-0.246,-0.249,-0.28,-0.297,-0.239,-0.179],

                'Compact shape':[-0.073,-0.073,-0.064,-0.065,-0.073,-0.078,-0.063,-0.047],
                'Normal shape':[-0.019,-0.019,-0.017,-0.017,-0.019,-0.02,-0.016,-0.012],
                'Slender shape':[0,0,0,0,0,0,0,0],

                'Refitting': [0.004,0.004,0.003,0.003,0.004,0.004,0.003,0.002],
                'Delivery': [-0.015,-0.015,-0.013,-0.013,-0.015,-0.016,-0.013,-0.008],
                'Racing': [-0.013,-0.013,-0.012,-0.012,-0.013,-0.014,-0.011,-0.008],
                'Long distance travel': [0.029,0.029,0.025,0.026,0.029,0.031,0.025,0.016],
                'Cross-country': [-0.023,-0.023,-0.02,-0.021,-0.023,-0.025,-0.02,-0.013],
                'Pick up children': [-0.006,-0.006,-0.005,-0.005,-0.006,-0.006,-0.005,-0.003],
                'Online car hailing': [0.046,0.046,0.04,0.041,0.046,0.049,0.039,0.026],
                'Commute': [0.031,0.031,0.027,0.028,0.031,0.033,0.027,0.018],
                'Fleet': [0.04,0.04,0.035,0.036,0.04,0.043,0.034,0.022],
                'Appointment': [0.032,0.032,0.028,0.029,0.032,0.034,0.028,0.018],
                'Trip': [0.006,0.006,0.005,0.005,0.006,0.006,0.005,0.003],
                'Business': [-0.004,-0.004,-0.003,-0.003,-0.004,-0.004,-0.003,-0.002],
                'Shopping':[0.038,0.038,0.033,0.034,0.038,0.04,0.032,0.021],

                '100000': [0.031,0.028,0.063,0.021,0.035,0.025,0.024,0.359],
                '100000-200000': [0.03,0.026,0.06,0.02,0.033,0.024,0.023,0.343],
                '200000-300000': [0.02,0.018,0.041,0.013,0.023,0.016,0.015,0.233],
                'Above 300000': [0, 0, 0, 0, 0, 0, 0, 0],
                'Fourth-tier city': [0, 0, 0, 0, 0, 0, 0, 0],
                'First-tier city': [0.02,0.018,0.04,0.013,0.022,0.016,0.015,0.229],
                'Second-tier city': [0.015,0.013,0.03,0.01,0.017,0.012,0.011,0.173],
                'Third-tier city': [0.028,0.024,0.055,0.018,0.031,0.022,0.021,0.316],
            }
            oil_lg,oil_tx,oil_qdd=lungu.get(lungu.curselection()),tixing.get(tixing.curselection()),qddeng.get(qddeng.curselection())
            oil_wd,oil_gs=weideng.get(weideng.curselection()),geshan.get(geshan.curselection())
            oil_gml,oil_cs=money.get(money.curselection()),city.get(city.curselection())
            oil_gcmd=[goal.get(i) for i in goal.curselection()]
            # 计算每个意象的得分
            if c.get(c.curselection())=='Fuel':valuedict=ry_dict
            else:valuedict=xny_dict
            select=[valuedict[oil_lg+' wheel-hub'], valuedict[oil_tx+' shape'],valuedict[oil_qdd+' headlight'],
                    valuedict[oil_wd+' taillight'], valuedict[oil_gs+' grille'],
                    valuedict[oil_gml],valuedict[oil_cs+' city']]
            if type(oil_gcmd)==str:select.append(valuedict[oil_gcmd])
            else:select.extend([valuedict[i] for i in oil_gcmd])

            final_oil = np.sum(select,axis=0)
            final_oil = [np.exp(i) for i in final_oil]
            final_oil = final_oil / sum(final_oil)
            final_oil = [i * 50 for i in final_oil]

            establish_leida(final_oil)
            # 将绘制而成的雷达图放在应用界面上



            global photo,imglabel
            img = Image.open('leida.png')  # 打开图片
            photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
            # img_canvas.create_image(155, 150, image=photo)
            imglabel = tk.Label(canvas2, image=photo,borderwidth=0)
            imglabel.place(x=570, y=350)

        '''匹配车款部分'''
        def pi_function():
            if lungu.get(lungu.curselection()) and tixing.get(tixing.curselection()):find=True
            else:find=False
            '''基本信息'''
            if find:
                base_canvas = tk.Canvas(master=canvas2, width=240, height=180, highlightthickness=0, borderwidth=0,
                                        highlightcolor='white', bg='white')
                base_canvas.place(x=915, y=60)
                base_canvas.create_text(60, 20, text='Information', font=('黑体', 11, 'bold'), fill='DarkSlateBlue')
                base_canvas.create_rectangle(10, 30, 120, 32, fill='DarkSlateBlue', outline='DarkSlateBlue')
                base_text = tk.Text(base_canvas, width=60, height=20, font=('黑体', 11), fg='black', borderwidth=0)
                top_k = ['Name： Xiaopeng P7\n', 'Price：278.8K\n', 'Tyep：New energy\n', 'Size：4880*1896*1450\n',
                         'Wheel-hub：Double five-spoke\n','Headlight：Rectangle\n','grille：None\n','Shape：Slender\n','Taillight：Pulling-through']
                for i in range(1, len(top_k) + 1):
                    base_text.insert(float(i), top_k[i - 1])
                base_text.place(x=10, y=38)

                '''评分信息'''
                score_canvas = tk.Canvas(master=canvas2, width=200, height=180, highlightthickness=0, borderwidth=0,
                                        highlightcolor='white', bg='white')
                score_canvas.place(x=1145, y=60)
                score_canvas.create_text(60, 20, text='User score', font=('黑体', 11, 'bold'), fill='DarkSlateBlue')
                score_canvas.create_rectangle(10, 30, 200, 32, fill='DarkSlateBlue', outline='DarkSlateBlue')
                score_text = tk.Text(score_canvas, width=70, height=20, font=('黑体', 11), fg='black', borderwidth=0)
                top_k = ['Exterior：   ★★★★★ 5\n', 'Interior：   ★★★★★ 5\n',
                         'Space：      ★★★★☆ 4\n','Power：      ★★★★★ 5\n',
                         'Operation：  ★★★★★ 5\n', 'Comfort：    ★★★★★ 5\n']
                for i in range(1, len(top_k) + 1):
                    score_text.insert(float(i), top_k[i - 1])
                score_text.place(x=15, y=38)

                #放置匹配到的汽车图片
                global photo1
                app1 = Image.open(u'正视图.png')  # 打开图片
                photo1 = ImageTk.PhotoImage(app1)  # 用PIL模块的PhotoImage打开
                imglabel1 = tk.Label(canvas2, image=photo1,borderwidth=0)
                imglabel1.place(x=930, y=280)

                global photo2
                app2 = Image.open(u'斜视图.png')  # 打开图片
                photo2 = ImageTk.PhotoImage(app2)  # 用PIL模块的PhotoImage打开
                imglabel2 = tk.Label(canvas2, image=photo2,borderwidth=0)
                imglabel2.place(x=1150, y=280)

                global photo3
                app3 = Image.open(u'侧视图.png')  # 打开图片
                photo3 = ImageTk.PhotoImage(app3)  # 用PIL模块的PhotoImage打开
                imglabel3 = tk.Label(canvas2, image=photo3,borderwidth=0)
                imglabel3.place(x=930, y=450)

                global photo4
                app4 = Image.open(u'后视图.png')  # 打开图片
                photo4 = ImageTk.PhotoImage(app4)  # 用PIL模块的PhotoImage打开
                imglabel4 = tk.Label(canvas2, image=photo4,borderwidth=0)
                imglabel4.place(x=1150, y=450)



        '''计算按钮'''
        cal_button=tk.Button(master=canvas2,text='Calculate',font = ('Times new roman',12,'bold'),command=cal_function)
        cal_button.place(x=620,y=620)
        '''匹配按钮'''
        pi_button=tk.Button(master=canvas2,text='Match',font = ('Times new roman',12,'bold'),command=pi_function)
        pi_button.place(x=770,y=620)

    else:
        canvas3=tk.Canvas(master=window,width=1400, height=700, highlightthickness=0, borderwidth=0,bg='Lavender')
        canvas3.place(x=200,y=0)

        '''车款资料'''
        base_canvas = tk.Canvas(master=canvas3, width=500, height=340, highlightthickness=0, borderwidth=0, bg='white')
        base_canvas.place(x=20, y=70)
        base_canvas.create_rectangle(6, 10, 10, 30, fill='DarkSlateblue', outline='DarkSlateblue')
        base_canvas.create_text(60, 20, text='Information', font=('黑体', 12), fill='black', )

        '''意象图对比'''
        compare_canvas = tk.Canvas(master=canvas3, width=450, height=300, highlightthickness=0, borderwidth=0,
                                   bg='white')
        compare_canvas.place(x=530, y=70)
        compare_canvas.create_rectangle(6, 10, 10, 30, fill='DarkSlateblue', outline='DarkSlateblue')
        compare_canvas.create_text(70, 20, text='Kansei Images', font=('黑体', 12), fill='black', )

        '''销量信息'''
        sale_canvas = tk.Canvas(master=canvas3, width=500, height=260, highlightthickness=0, borderwidth=0,
                                    bg='white')
        sale_canvas.place(x=20, y=420)
        sale_canvas.create_rectangle(6, 10, 10, 30, fill='DarkSlateblue', outline='DarkSlateblue')
        sale_canvas.create_text(60, 20, text='Sale status', font=('黑体', 12), fill='black', )

        '''图片'''
        image_canvas = tk.Canvas(master=canvas3, width=450, height=300, highlightthickness=0, borderwidth=0, bg='white')
        image_canvas.place(x=530, y=380)
        image_canvas.create_rectangle(6, 10, 10, 30, fill='DarkSlateblue', outline='DarkSlateblue')
        image_canvas.create_text(75, 20, text='Vehicle images', font=('黑体', 12), fill='black', )

        '''车辆评估'''
        evaluate_canvas = tk.Canvas(master=canvas3, width=390, height=610, highlightthickness=0, borderwidth=0, bg='white')
        evaluate_canvas.place(x=990, y=70)
        evaluate_canvas.create_rectangle(6, 10, 10, 30, fill='DarkSlateblue', outline='DarkSlateblue')
        evaluate_canvas.create_text(60, 20, text='Evaluation', font=('黑体', 12), fill='black', )





        canvas4=tk.Canvas(master=canvas3,width=500,height=40, highlightthickness=0, borderwidth=0,bg='white')
        canvas4.place(x=20,y=20)
        canvas4.create_rectangle(6,10,10,30,fill='DarkSlateblue',outline='DarkSlateblue')
        canvas4.create_text(82,20,text='Vehicle：　　　',font = ('黑体',12),fill='black',)
        input_car=tk.Entry(master=canvas4,font = ('黑体',12),width=14,bd =2)
        input_car.place(x=100,y=10)

        """生成报告函数"""

        def report():
            search_car=input_car.get()#获取输入的查询车辆
            if search_car:
                '''显示基本信息'''

                base_canvas1 = tk.Canvas(master=base_canvas, width=240, height=180, highlightthickness=0, borderwidth=0,
                                         highlightcolor='white', bg='white')
                base_canvas1.place(x=40, y=40)
                base_canvas1.create_text(55, 20, text='Information', font=('黑体', 11, 'bold'), fill='DarkSlateBlue')
                base_canvas1.create_rectangle(5, 30, 120, 32, fill='DarkSlateBlue', outline='DarkSlateBlue')
                base_text1 = tk.Text(base_canvas1, width=50, height=24, font=('黑体', 11), fg='black', borderwidth=0)
                top_k1 = ['Name：XiaoPeng P7\n', 'Type：New energy\n', 'Size：4880*1896*1450\n',
                          'Wheel-hub：Double five-spoke\n', 'Headlight：Rectangle\n', 'grille：None\n', 'Shape：Slender\n', 'Taillight：Pulling-through\n',
                          'Price：278.8K\n']
                for i in range(1, len(top_k1) + 1):
                    base_text1.insert(float(i), top_k1[i - 1])
                base_text1.place(x=5, y=38)

                '''评分信息'''
                score_canvas1 = tk.Canvas(master=base_canvas, width=200, height=180, highlightthickness=0,
                                          borderwidth=0,
                                          highlightcolor='white', bg='white')
                score_canvas1.place(x=280, y=40)
                score_canvas1.create_text(60, 20, text='User score', font=('黑体', 11, 'bold'), fill='DarkSlateBlue')
                score_canvas1.create_rectangle(10, 30, 200, 32, fill='DarkSlateBlue', outline='DarkSlateBlue')
                score_text1 = tk.Text(score_canvas1, width=60, height=24, font=('黑体', 11), fg='black', borderwidth=0)
                top_k1 = ['Exterior：   ★★★★★ 5\n', 'Interior：   ★★★★★ 5\n',
                          'Space：      ★★★★☆ 4\n', 'Power：      ★★★★★ 5\n',
                          'Operation：  ★★★★★ 5\n', 'Comfort：    ★★★★★ 5\n', ]
                for i in range(1, len(top_k1) + 1):
                    score_text1.insert(float(i), top_k1[i - 1])
                score_text1.place(x=15, y=38)

                '''口碑印象'''
                koubei_canvas = tk.Canvas(master=base_canvas, width=500, height=100, highlightthickness=0,
                                          borderwidth=0,
                                          highlightcolor='white', bg='white')
                koubei_canvas.place(x=50, y=220)
                koubei_canvas.create_text(60, 20, text='Expression', font=('黑体', 11, 'bold'), fill='DarkSlateBlue')
                koubei_canvas.create_rectangle(10, 30, 350, 32, fill='DarkSlateBlue', outline='DarkSlateBlue')

                koubei_canvas.create_rectangle(10, 40, 150, 60, fill='Lavender', outline='Lavender')
                koubei_canvas.create_text(80, 50, text='Strong acceleration(75)', fill='DimGrey')
                koubei_canvas.create_rectangle(160, 40, 300, 60, fill='Pink', outline='Pink')
                koubei_canvas.create_text(230, 50, text='High configuration(100)', fill='DimGrey')
                # koubei_canvas.create_rectangle(210, 40, 300, 60, fill='LightGreen', outline='LightGreen')
                # koubei_canvas.create_text(255, 50, text='High configuration(96)', fill='DimGrey')
                koubei_canvas.create_rectangle(10, 70, 110, 90, fill='Magenta', outline='Magenta')
                koubei_canvas.create_text(60, 80, text='Big mileage(167)', fill='DimGrey')
                koubei_canvas.create_rectangle(120, 70, 230, 90, fill='indianRed', outline='indianRed')
                koubei_canvas.create_text(175, 80, text='Good interior(55)', fill='DimGrey')

                global photo6
                app6 = Image.open(u'销量图.png')  # 打开图片
                photo6 = ImageTk.PhotoImage(app6)  # 用PIL模块的PhotoImage打开
                imglabel4 = tk.Label(sale_canvas, image=photo6, borderwidth=0)
                imglabel4.place(x=30, y=30)

                global photo7
                img7 = Image.open('leida.png')  # 打开图片
                photo7 = ImageTk.PhotoImage(img7)  # 用PIL模块的PhotoImage打开
                imglabel7 = tk.Label(compare_canvas, image=photo7, borderwidth=0)
                imglabel7.place(x=70, y=40)

                global photo8
                img8 = Image.open('评估图.jpg')  # 打开图片
                photo8 = ImageTk.PhotoImage(img8)  # 用PIL模块的PhotoImage打开
                imglabel8 = tk.Label(evaluate_canvas, image=photo8, borderwidth=0)
                imglabel8.place(x=40, y=60)

                global photo9
                img9 = Image.open('正视图.png')  # 打开图片
                photo9 = ImageTk.PhotoImage(img9)  # 用PIL模块的PhotoImage打开
                imglabel9 = tk.Label(image_canvas, image=photo9, borderwidth=0)
                imglabel9.place(x=20, y=80)

                global photo10
                img10 = Image.open('侧视图.png')  # 打开图片
                photo10 = ImageTk.PhotoImage(img10)  # 用PIL模块的PhotoImage打开
                imglabel10 = tk.Label(image_canvas, image=photo10, borderwidth=0)
                imglabel10.place(x=230, y=80)

            return


        car_button=tk.Button(master=canvas4,font = ('黑体',9),text='Generate',command=report)
        car_button.place(x=240,y=10)
        canvas4.create_rectangle(30, 380, 640, 382, fill='LightGrey', outline='LightGrey')
        canvas4.create_rectangle(30, 390, 640, 392, fill='LightGrey', outline='LightGrey')


user.bind("<Double-Button-1>", application_select)


window.mainloop()
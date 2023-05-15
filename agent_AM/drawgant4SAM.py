import time
import plotly as py
import plotly.figure_factory as ff
from agent_AM.problems.schedule.args import orders_number

is_text = False  # 是否在上面加解释

# (开始时间，持续时间（服务时间），公司enterprise编号，订单order编号)
one_process = [
(0.0,1.81,18,0, 0.88),(6.23,1.79,2,0, 0.42),(10.1,1.71,15,0, 1.03),(16.96,0.68,17,0, 0.42),(19.74,2.41,10,0, 0.38),(24.07,2.22,3,0, 0.33),(27.94,1.99,16,0, 0.27),(31.26,1.24,15,0, 0.68),(35.88,1.29,13,0, 0.76),(40.97,1.86,15,0, 0.76),(46.63,1,0,0, 0.25),(0.0,1.86,15,0, 0.25),(3.11,1.94,13,1, 0.76),(8.84,1.82,20,1, 0.09),(11.14,1.71,15,1, 0.73),(16.52,2.41,10,1, 0.61),(21.97,1.83,9,1, 0.35),(25.56,2.22,3,1, 0.55),(30.54,1.21,18,1, 0.32),(33.34,0.83,6,1, 0.29),(35.6,2.05,7,1, 0.39),(39.6,1,0,1, 0.37),(0.0,1.94,13,1, 0.96),(6.74,1.82,8,2, 0.62),(11.64,1.78,14,2, 0.7),(16.9,2.28,8,2, 0.7),(22.65,1.12,5,2, 0.17),(24.62,2.2,12,2, 0.4),(28.82,0.91,20,2, 0.45),(31.98,1.86,15,2, 0.73),(37.51,2.27,18,2, 0.63),(42.95,1.94,13,2, 0.44),(47.1,1,0,2, 0.96),(0.0,1.56,1,2, 0.65),(4.82,1.56,1,3, 0.0),(6.38,1.51,7,3, 0.34),(9.61,2.42,13,3, 0.69),(15.49,2.05,7,3, 0.69),(21.0,2.44,9,3, 0.68),(26.86,2.28,8,3, 0.72),(32.76,1.9,3,3, 0.74),(38.38,1.28,10,3, 0.33),(41.32,1.61,11,3, 0.49),(45.36,1,0,3, 0.96),(0.0,1.29,13,3, 0.96),(6.09,1.99,16,4, 0.14),(8.76,1.61,11,4, 0.78),(14.29,0.94,11,4, 0.0),(15.23,2.14,9,4, 0.24),(18.56,1.62,17,4, 0.41),(22.26,1.02,2,4, 0.8),(27.29,1.83,9,4, 0.55),(31.87,0.83,6,4, 0.3),(34.2,2.23,14,4, 0.86),(40.74,1,0,4, 0.99),(0.0,1.08,17,4, 0.65),(4.32,1.84,4,5, 0.35),(7.89,1.62,17,5, 0.35),(11.24,1.02,2,5, 0.8),(16.28,1.99,16,5, 0.44),(20.46,2.01,11,5, 0.78),(26.39,1.77,13,5, 0.92),(32.76,1.81,18,5, 0.44),(36.79,2.14,9,5, 0.33),(40.59,2.14,9,5, 0.0),(42.73,1,0,5, 1.05),(0.0,2.29,9,5, 1.05),(7.53,1.82,19,6, 0.76),(13.17,2.01,11,6, 0.75),(18.95,2.48,6,6, 0.27),(22.78,1.28,10,6, 0.28),(25.47,1.64,7,6, 0.49),(29.58,2.25,5,6, 0.08),(32.21,1.12,10,6, 0.46),(35.63,2.27,18,6, 0.03),(38.03,1.22,9,6, 0.33),(40.92,1,0,6, 1.05),(0.0,1.54,5,6, 0.43),(3.7,1.56,20,7, 0.71),(8.79,1.09,7,7, 0.68),(13.29,1.56,1,7, 0.34),(16.57,2.27,18,7, 0.31),(20.37,1.99,16,7, 0.31),(23.9,2.29,9,7, 0.64),(29.38,1.84,4,7, 0.15),(31.95,2.17,15,7, 0.77),(37.96,1.94,13,7, 0.76),(43.69,1,0,7, 0.96),(0.0,0.74,14,7, 0.99),(5.69,1.09,15,8, 0.83),(10.91,1.82,20,8, 0.73),(16.4,1.62,17,8, 0.78),(21.94,1.56,20,8, 0.78),(27.41,1.99,16,8, 0.2),(30.4,2.14,9,8, 0.64),(35.73,1.08,17,8, 0.41),(38.88,1.93,10,8, 0.38),(42.73,2.44,9,8, 0.35),(46.92,1,0,8, 1.05),(0.0,2.36,12,8, 0.47),(4.72,0.8,10,9, 0.55),(8.29,2.05,2,9, 0.44),(12.52,2.14,4,9, 0.69),(18.13,1.54,2,9, 0.69),(23.13,1.06,8,9, 0.89),(28.66,1.22,4,9, 0.71),(33.41,1.92,2,9, 0.69),(38.8,1.71,15,9, 1.03),(45.65,1.97,5,9, 0.19),(48.58,1,0,9, 0.43),(0.0,2.23,14,9, 0.99),(7.17,1.28,10,10, 0.63),(11.6,2.2,12,10, 0.55),(16.57,1.64,7,10, 0.34),(19.89,1.47,11,10, 0.63),(24.49,1.19,14,10, 1.11),(31.22,0.94,11,10, 1.11),(37.71,1.86,15,10, 0.74),(43.26,1.62,17,10, 0.42),(46.99,0.8,10,10, 0.38),(49.71,1,0,10, 0.86),(0.0,1.69,5,10, 0.43),(3.84,1.78,14,11, 0.82),(9.72,0.76,4,11, 1.06),(15.78,2.13,8,11, 0.71),(21.43,1.62,17,11, 0.38),(24.97,1.54,2,11, 0.8),(30.52,2.49,16,11, 0.44),(35.2,2.05,2,11, 0.44),(39.44,2.27,18,11, 0.42),(43.79,1.09,7,11, 0.52),(47.47,1,0,11, 0.37),(0.0,1.54,5,11, 0.43),(3.7,1.92,2,12, 0.89),(10.08,1.94,13,12, 0.49),(14.46,2.29,4,12, 0.86),(21.06,0.76,18,12, 0.42),(23.92,1.9,3,12, 0.32),(27.41,1.43,20,12, 0.42),(30.92,2.05,2,12, 0.58),(35.89,1.09,15,12, 1.03),(42.12,1.92,7,12, 0.12),(44.66,1,0,12, 0.37),(0.0,1.94,13,12, 0.96),(6.74,2.01,11,13, 0.92),(13.34,1.83,9,13, 0.24),(16.38,1.69,5,13, 0.62),(21.16,2.14,4,13, 0.57),(26.17,2.45,4,13, 0.0),(28.62,1.22,9,13, 0.15),(30.58,1.82,20,13, 0.84),(36.58,2.41,10,13, 0.49),(41.45,1.82,8,13, 0.48),(45.69,1,0,13, 0.38),(0.0,2.01,11,13, 0.96),(6.79,1.12,10,14, 0.49),(10.35,1.83,9,14, 0.35),(13.93,1.21,18,14, 0.33),(16.81,2.49,16,14, 0.31),(20.84,2.33,15,14, 0.68),(26.55,1.56,1,14, 0.43),(30.24,2.45,4,14, 0.65),(35.95,2.14,9,14, 0.15),(38.82,2.28,8,14, 0.72),(44.71,1,0,14, 0.38),(0.0,1.22,4,14, 0.99),(6.19,1.82,8,15, 0.71),(11.54,1.07,9,15, 0.72),(16.23,1.61,11,15, 0.24),(19.03,2.19,7,15, 0.63),(24.35,2.28,8,15, 0.1),(27.15,1.9,3,15, 0.74),(32.78,1.63,14,15, 0.52),(37.01,1.62,17,15, 0.91),(43.19,0.76,4,15, 0.35),(45.69,1,0,15, 0.99),(0.0,1.95,1,15, 0.65),(5.21,2.29,4,16, 0.65),(10.77,1.62,17,16, 0.35),(14.12,1.04,14,16, 0.91),(19.72,0.8,10,16, 0.63),(23.67,1.54,2,16, 0.44),(27.38,2.26,13,16, 0.49),(32.08,2.16,17,16, 0.76),(38.02,1.61,11,16, 0.32),(41.22,1.26,12,16, 0.88),(46.9,1,0,16, 0.47),(0.0,1.94,13,16, 0.96),(6.74,1.98,6,17, 0.69),(12.16,1.78,14,17, 0.86),(18.26,1.19,14,17, 0.0),(19.44,1.62,17,17, 0.91),(25.63,1.99,16,17, 0.63),(30.75,2.65,16,17, 0.0),(33.4,1.95,1,17, 0.25),(36.61,1.93,10,17, 0.28),(39.94,2.41,10,17, 0.0),(42.35,1,0,17, 0.86),(0.0,1.24,15,17, 0.25),(2.49,1.82,16,18, 0.68),(7.7,1.92,2,18, 0.44),(11.81,1.79,2,18, 0.0),(13.61,1.56,20,18, 0.58),(18.08,1.04,20,18, 0.0),(19.12,1.64,7,18, 0.68),(24.17,2.05,2,18, 0.92),(30.82,2.27,18,18, 0.42),(35.16,1.62,17,18, 0.4),(38.77,1,0,18, 0.65),(0.0,2.42,13,18, 0.96),(7.22,1.51,7,19, 0.69),(12.19,1.04,20,19, 0.68),(16.63,1.62,17,19, 0.78),(22.17,2.23,14,19, 0.91),(28.96,1.09,15,19, 0.83),(34.18,1.54,2,19, 1.03),(40.87,1.69,5,19, 0.89),(47.02,2.29,4,19, 0.57),(52.19,2.41,10,19, 0.43),(56.75,1,0,19, 0.86),
]
title = "GANTT figure of SAM"

# 将one_process里的数据转化为四个列表
n_start_time = []
n_duration_time = []
n_bay_start = []
n_job_id = []
order_id = -1  # 不太好记录订单的变化，直接用0来看，因为边界一直在摇摆
for eve_tuple in one_process:

    n_start_time.append(int(eve_tuple[0]*100))
    n_duration_time.append(int(eve_tuple[1]*100))
    n_bay_start.append(int(eve_tuple[2]))
    if eve_tuple[0] == 0.0:
        order_id += 1
    n_job_id.append(order_id)

print(n_start_time,'\n',
n_duration_time,'\n',
n_bay_start,'\n',
n_job_id)
# print(n_bay_start)
# print(all_data)
# x轴, 对应于画图位置的起始坐标x
# start, time, of, every, task, , //每个工序的开始时间
# n_start_time = [0, 0, 2, 6, 0, 0, 3, 4, 10, 13, 4, 3, 10, 6, 12, 4, 5, 6, 14, 7, 9, 9, 16, 7, 11, 14, 15, 12, 16, 17,
#                 16, 15, 18, 19, 19, 20, 21, 20, 22, 21, 24, 24, 25, 27, 30, 30, 27, 25, 28, 33, 36, 33, 30, 37, 37, 40]
# # length, 对应于每个图形在x轴方向的长度
# # duration, time, of, every, task, , //每个工序的持续时间
# n_duration_time = [6, 2, 1, 6, 4, 3, 1, 6, 3, 3, 2, 1, 2, 1, 2, 1, 1, 3, 2, 2, 6, 2, 1, 4, 4, 2, 6, 6, 1, 2, 1, 4, 6, 1,
#                    6, 1, 1, 1, 5, 6, 1, 6, 4, 3, 6, 1, 6, 3, 2, 6, 1, 4, 6, 1, 5, 6]
#
# # y轴, 对应于画图位置的起始坐标y
# # bay, id, of, every, task, , ==工序数目，即在哪一行画线
# n_bay_start = [1, 5, 5, 1, 2, 4, 5, 5, 4, 4, 3, 0, 5, 2, 5, 0, 0, 3, 5, 0, 3, 0, 5, 2, 2, 0, 3, 1, 0, 5, 4, 2, 1, 0, 5,
#                0, 0, 2, 0, 3, 2, 1, 2, 0, 1, 0, 3, 4, 5, 3, 0, 2, 5, 2, 0, 6]
#
# # 工序号，可以根据工序号选择使用哪一种颜色
# # n_job_id = [1, 9, 8, 2, 0, 4, 6, 9, 9, 0, 6, 4, 7, 1, 5, 8, 3, 8, 2, 1, 1, 8, 9, 6, 8, 5, 8, 4, 2, 0, 6, 7, 3, 0, 2, 1, 7, 0, 4, 9, 3, 7, 5, 9, 5, 2, 4, 3, 3, 7, 5, 4, 0, 6, 5]
# n_job_id = ['B', 'J', 'I', 'C', 1, 'E', 'G', 'J', 'J', 1, 'G', 'E', 'H', 'B', 'F', 'I', 'D', 'I', 'C', 'B', 'B',
#             'I', 'J', 'G', 'I', 'F', 'I', 'E', 'C', 1, 'G', 'H', 'D', 1, 'C', 'B', 'H', 1, 'E', 'J', 'D', 'H',
#             'F', 'J', 'F', 'C', 'E', 'D', 'D', 'H', 'F', 'E', 1, 'G', 'F', "F"]

print(len(n_bay_start), len(n_job_id))

# belows are the number of planes
op = [i for i in range(orders_number)]
# print(op)

cnames = {
'aliceblue':            '#F0F8FF',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'crimson':              '#DC143C',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

colors = tuple(cnames.values())

millis_seconds_per_minutes = 1000 * 60
start_time = time.time() * 1000

job_sumary = {}


# 获取工件对应的第几道工序
def get_op_num(job_num):
    index = job_sumary.get(str(job_num))
    new_index = 1
    if index:
        new_index = index + 1
    job_sumary[str(job_num)] = new_index
    return new_index


def create_draw_defination():
    df = []
    for index in range(len(n_job_id)):
        operation = {}
        # 机器，纵坐标
        if n_bay_start.__getitem__(index) == 0:
            operation['Task'] = 'Loc 0'
        else:
            operation['Task'] = 'Loc ' + str(n_bay_start.__getitem__(index))
        operation['Start'] = start_time.__add__(n_start_time.__getitem__(index) * millis_seconds_per_minutes)
        operation['Finish'] = start_time.__add__(
            (n_start_time.__getitem__(index) + n_duration_time.__getitem__(index)) * millis_seconds_per_minutes)
        # 工件，
        job_num = op.index(n_job_id.__getitem__(index)) + 1
        operation['Resource'] = 'P' + str(job_num)
        df.append(operation)
        # print(operation['Task'])
    df.sort(key=lambda x: int(x["Task"][4:]), reverse=True)
    return df


def draw_prepare():
    df = create_draw_defination()
    return ff.create_gantt(df, colors=colors, index_col='Resource',
                           title=title, show_colorbar=True,
                           group_tasks=True, data=n_duration_time,
                           showgrid_x=False, showgrid_y=True)


def add_annotations(fig):
    y_pos = 0
    for index in range(len(n_job_id)):
        # 机器，纵坐标
        y_pos = n_bay_start.__getitem__(index)

        x_start = start_time.__add__(n_start_time.__getitem__(index) * millis_seconds_per_minutes)
        # a= start_time.__add__(16)
        # print(type(start_time), x_start, type(n_start_time.__getitem__(index)),index,a)
        x_end = start_time.__add__(
            (n_start_time.__getitem__(index) + n_duration_time.__getitem__(index)) * millis_seconds_per_minutes)
        x_pos = (x_end - x_start) / 2 + x_start

        # 工件，
        job_num = op.index(n_job_id.__getitem__(index)) + 1
        if is_text:
            text = 'P(' + str(job_num) + "," + str(get_op_num(job_num)) + ")=" + str(n_duration_time.__getitem__(index))
            # text = 'T' + str(job_num) + str(get_op_num(job_num))
        else:
            text = ''
        text_font = dict(size=14, color='black')
        # print(x_pos, y_pos, text)
        fig['layout']['annotations'] += tuple(
            [dict(x=x_pos, y=y_pos+0.2, text=text, textangle=-15, showarrow=False, font=text_font)])


def draw_fjssp_gantt():
    fig = draw_prepare()
    add_annotations(fig)
    py.offline.plot(fig, filename='SAM-GANTT.html')


if __name__ == '__main__':
    draw_fjssp_gantt()

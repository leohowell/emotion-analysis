import csv
import io
import json
import os
import time

import numpy as np
import requests


class JdSpider:
    headers = {
        'User-Agent': 'Mozilla/4.0 (Windows NT 10.0; WOW64) AppleWebKit/521.36 (XHTML, like Gecko) '
                      'Chrome/89.0.3325.181 Safari/537.36'}

    base_product_url = 'https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv158&' \
                       'productId={0}&score={1}&sortType=5&page={2}&pageSize=10&isShadowSku=0&fold=1'
    list_comment = []
    pro_id_list = []

    root = '../dataset/reviews'

    def __init__(self, pro_id_list):
        self.pro_id_list = pro_id_list

    # 保存评论数据
    def comment_to_csv(self, file_name):
        if not os.path.exists(JdSpider.root):
            os.makedirs(JdSpider.root)
        path = '%s/%s.csv' % (JdSpider.root, file_name)
        file = io.open(path, 'a+', encoding="utf-8", newline='')
        writer = csv.writer(file)
        if not os.path.getsize(path):
            # content, creationTime, nickname, referenceName, content_type
            writer.writerow(
                ['creationTime', 'nickname', 'content', 'buy_time', '点赞数', '回复数', 'content_type', 'referenceName'])
        for index in range(len(self.list_comment)):
            writer.writerow(self.list_comment[index])
        file.close()
        print('写入%d条---文件%s.csv' % (len(self.list_comment), file_name))

    def get_comment_data(self, proc_id, index, max_page):
        comment = []
        cur_page = 0
        while cur_page < max_page:
            cur_page += 1
            # if (cur_page > 3):  # 测试用代码
            #     break
            # noinspection PyBroadException
            try:
                comment_url = JdSpider.base_product_url.format(proc_id, index, cur_page)
                comment_json = requests.get(url=comment_url, headers=JdSpider.headers).text
                time.sleep(2)
                # print(jsonData[::-1])//字符串逆序
                comment_json = comment_json[comment_json.find('{'):-2]
                comment_json = json.loads(comment_json)
                page_len = len(comment_json['comments'])
                print("当前第%s页---url=%s" % (cur_page, comment_url))
                self.list_comment = []
                for j in range(0, page_len):
                    user_id = comment_json['comments'][j]['id']  # 用户ID
                    content = comment_json['comments'][j]['content']  # 评论内容
                    bought_time = comment_json['comments'][j]['referenceTime']  # 购买时间
                    vote_count = comment_json['comments'][j]['usefulVoteCount']  # 点赞数
                    reply_count = comment_json['comments'][j]['replyCount']  # 回复数目
                    star_step = comment_json['comments'][j]['score']  # 得分
                    creation_time = comment_json['comments'][j]['creationTime']  # 评价时间
                    reference_name = comment_json['comments'][j]['referenceName']  # 商品名字
                    comment.append(creation_time)
                    comment.append(user_id)  # 每一行数据
                    comment.append(content)
                    comment.append(bought_time)
                    comment.append(vote_count)
                    comment.append(reply_count)
                    comment.append(star_step)
                    comment.append(reference_name)
                    self.list_comment.append(comment)
                    comment = []
                self.comment_to_csv(proc_id)
            except Exception as e2:
                print("the error2 is ", e2)
                time.sleep(5)
                cur_page -= 1
                print('网络故障或者是网页出现了问题，五秒后重新连接')

    def get_jd_comment(self):
        if self.pro_id_list is None:
            return
        for proc in self.pro_id_list:  # 遍历产品列表
            i = -1
            while i < 7:  # 遍历排序方式
                i += 1
                path = '%s/%s.csv' % (JdSpider.root, proc)
                if os.path.exists(path):
                    continue
                self.comment_to_csv(proc)
                # 先访问第0页获取最大页数，再进行循环遍历
                root_url = JdSpider.base_product_url.format(proc, i, 0)
                try:
                    json_data = requests.get(url=root_url, headers=JdSpider.headers).text
                    # print("jsonData---%s" % json_data)
                    start_loc = json_data.find('{')
                    json_data = json_data[start_loc:-2]
                    json_data = json.loads(json_data)
                    print("最大页数%s" % json_data['maxPage'])
                    self.get_comment_data(proc, i, json_data['maxPage'])  # 遍历每一页
                except Exception as e:
                    i -= 1
                    print("the error is ", e)
                    time.sleep(5)


if __name__ == "__main__":
    # 这里写商品id，有时候连接失败，可能需要打开京东网页再试
    # jd = JdSpider(['3914278'])
    jd = JdSpider(['100007627021'])
    jd.get_jd_comment()

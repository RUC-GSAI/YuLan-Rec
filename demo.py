import cv2
import time
import gradio as gr
from typing import Dict, List

from simulator import Simulator
from utils.message import Message
from utils.utils import layout_img, get_avatar1, html_format, chat_format, rec_format, social_format, round_format


class Demo:
    """
    the UI configurations of Demonstration
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.simulator = Simulator(config, logger)
        self.round=0
        self.cur_image = None
        self.cur_log = ""
        self.cur_chat = ""
        self.cur_rec = ""
        self.cur_post = ""
        self.cur_round = ""
        self.play = False
        self.sleep_time = 3
        self.css_path = './asset/css/styles.css'
        self.init_round_info = '<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 20px; color: #000000; font-weight: bold;">&nbsp;&nbsp; Waiting to start !  &nbsp;&nbsp;</div>'


    def init_background(self):
        background = cv2.imread("./asset/img/background1.png")
        back_h, back_w, _ = background.shape

        small_height_list = [350, 130, 130, 130,
                             350, 130, 350, 350,
                             520, 520, 520,
                             720, 720, 720, 720,
                             900, 900, 900, 900, 900]
        small_weight_list = [700, 500, 850, 1300,
                             350, 150, 1000, 1400,
                             100, 500, 1000,
                             350, 750, 1200, 1500,
                             200, 500, 850, 1050, 1300]

        small_coordinate = list(zip(small_height_list, small_weight_list))
        for id in range(20):
            img = cv2.imread(f"./asset/img/s_{id}.png", cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, dsize=None, fx=0.063, fy=0.063)  # 0.063  0.1
            layout_img(background, img, small_coordinate[id])

        return background

    def reset(self):
        # reset simulator
        del self.simulator
        self.simulator=Simulator(self.config, self.logger)
        self.simulator.load_simulator()
        # reset image and text
        background = self.init_background()
        return [cv2.cvtColor(background, cv2.COLOR_BGR2RGB), "", "", "", "", self.init_round_info]

    def format_message(self, messages: List[Message]):
        _format = [
            {"original_content": "", "content": "", "agent_id": messages[idx].agent_id, "action": messages[idx].action,
             "msg_id": idx}
            for idx in range(len(messages))]

        for idx in range(len(messages)):
            _format[idx]["original_content"] = "[{}]: {}".format(self.agent_dict[messages[idx].agent_id],
                                                                 messages[idx].content)
            _format[idx]["content"] = html_format(messages[idx].content)

        return _format

    def generate_img_once(self, data: List[Dict]):
        background = self.init_background()
        big_height_list = [300, 80, 80, 80,     # small-50
                           300, 80, 300, 300,
                           470, 470, 470,
                           670, 670, 670, 670,
                           850, 850, 850, 850, 850]
        big_weight_list = [670, 470, 820, 1270,    # small-30
                           320, 120, 970, 1370,
                           70, 470, 970,
                           320, 720, 1170, 1470,
                           170, 470, 820, 1020, 1270]

        icon_height_list = [280, 60, 60, 60,     # big-20
                           280, 60, 280, 280,
                           450, 450, 450,
                           650, 650, 650, 650,
                           830, 830, 830, 830, 830]
        icon_weight_list = [790, 590, 940, 1390,    # big+120
                           440, 240, 1090, 1490,
                           190, 590, 1090,
                           440, 840, 1290, 1590,
                           290, 590, 940, 1140, 1390]

        big_coordinate = list(zip(big_height_list, big_weight_list))
        icon_coordinate = list(zip(icon_height_list, icon_weight_list))

        for idx in range(len(data)):
            img = cv2.imread("./asset/img/b_{}.png".format(data[idx]['agent_id']), cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
            layout_img(background, img, big_coordinate[data[idx]['agent_id']])

            if data[idx]['action'] == 'RECOMMENDER':
                img = cv2.imread("./asset/img/recsys.png", cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
                layout_img(background, img, icon_coordinate[data[idx]['agent_id']])
            elif data[idx]['action'] == 'POST':
                img = cv2.imread("./asset/img/social.png", cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
                layout_img(background, img, icon_coordinate[data[idx]['agent_id']])
            elif data[idx]['action'] == 'CHAT':
                img = cv2.imread("./asset/img/chat.png", cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
                layout_img(background, img, icon_coordinate[data[idx]['agent_id']])

        return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)


    def generate_text_once(self, data: List[Dict], round: int):
        log = self.cur_log
        chat_log = ""
        rec_log = ""
        social_log = ""
        round_log = ""
        for msg in data:
            log += msg['content']
            # log += '\n\n'
            log += '<br><br>'
            if msg['action'] == 'CHAT':
                chat_log = chat_format(msg)
            elif msg['action'] == 'RECOMMENDER':
                rec_log = rec_format(msg)
            elif msg['action'] == 'POST':
                social_log = social_format(msg)
            round_log = round_format(round, self.agent_dict[msg['agent_id']])
        return log, chat_log, rec_log, social_log, round_log

    def generate_output(self):
        """
        generate new image and message of next step
        :return: [new image, new message]
        """
        self.round=self.round+1
        for i in range(self.agent_num):
            next_message = self.simulator.one_step(i)         
            data = self.format_message(next_message)
            for d in data:
                time.sleep(self.sleep_time)
                img = self.generate_img_once([d])
                log, chat_log, rec_log, social_log, round_log = self.generate_text_once([d], self.round)
                yield [img, log, chat_log, rec_log, social_log, round_log]

    def execute_reset(self):
        self.play = False
        self.cur_image, self.cur_log, self.cur_chat, self.cur_rec, self.cur_post, self.cur_round = self.reset()
        return self.cur_image, self.cur_log, self.cur_chat, self.cur_rec, self.cur_post, self.cur_round

    def execute_play(self):
        self.play = True
        while self.play:
            for output in self.generate_output():
                self.cur_image, self.cur_log, self.cur_chat, self.cur_rec, self.cur_post, self.cur_round = output
                if self.play:
                    yield self.cur_image, self.cur_log, self.cur_chat, self.cur_rec, self.cur_post, self.cur_round
                else:
                    return self.reset()
                time.sleep(self.sleep_time)

    def launch_demo(self):

        with gr.Blocks(theme="soft", title="RecAgent Demo", css=self.css_path) as demo:
            with gr.Row(variant='panel'):
                with gr.Column(scale=2, elem_classes=['column-container']):
                    background = self.init_background()
                    image_output = gr.Image(value=cv2.cvtColor(background, cv2.COLOR_BGR2RGB), label='Demo',
                                            show_label=False)
                    relation_output = gr.Image(value='./asset/img/relations2.png', label='Relations',
                                               show_label=False)
                with gr.Column(scale=1, elem_classes=['border', 'column-container']):

                    round_output = gr.HTML(value=self.init_round_info, elem_classes=['round'])

                    chat_pic = gr.HTML(value=get_avatar1('cha1'))
                    chat_output = gr.HTML(value="", show_label=False,
                                          elem_classes=['textbox_size', 'scrollable-textbox', 'textbox-font'], )

                    rec_pic = gr.HTML(value=get_avatar1('rec1'))
                    rec_output = gr.HTML(value="", show_label=False,
                                         elem_classes=['textbox_size', 'scrollable-textbox', 'textbox-font'], )

                    soc_pic = gr.HTML(value=get_avatar1('soc1'))
                    soc_output = gr.HTML(value="", show_label=False,
                                         elem_classes=['textbox_size', 'scrollable-textbox', 'textbox-font'], )

                    with gr.Row(variant='panel', elem_classes=['button-container']):
                        play_btn = gr.Button("Play", variant='primary',
                                             elem_id="play_btn", elem_classes=["btn_font", "btn_size"])
                        reset_btn = gr.Button("Reset", variant='primary',
                                              elem_id="reset_btn", elem_classes=["btn_font", "btn_size"])


            with gr.Row():
                log_pic = gr.HTML(value=get_avatar1('log1'), elem_classes=["log"])
            with gr.Row():
                log_output = gr.HTML(value="", show_label=False,
                                     elem_classes=['logbox_size', 'scrollable-textbox', 'textbox-font', 'border'])

            play_btn.click(fn=self.execute_play, inputs=None,
                           outputs=[image_output, log_output, chat_output, rec_output, soc_output, round_output],
                           show_progress=False)
            reset_btn.click(fn=self.execute_reset, inputs=None,
                            outputs=[image_output, log_output, chat_output, rec_output, soc_output, round_output],
                            show_progress=False)

        self.simulator.load_simulator()
        self.agent_num = len(self.simulator.agents.keys())
        self.agent_dict = {agent.id: agent.name for id,agent in self.simulator.agents.items()}

        demo.queue(concurrency_count=1, max_size=1).launch(height="100%", width="100%")

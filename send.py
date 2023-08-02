import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import mimetypes
import os
import time
import tarfile
import sys
import re
sys.path.append("../../bin")

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, 'w:gz') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def send_mail_ftp(sender, receiver, content_html,subject):

    # read html
    content = ''
    with open(content_html, 'r', encoding='utf-8') as f:
        for line in f:
            content += line

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ','.join(receiver)
    msg['subject'] = subject
    msg.attach(MIMEText(content, 'html', 'utf-8'))

    try:
        # login
        smtp = smtplib.SMTP_SSL('smtp.icoremail.net', 465)
        # need information
        smtp.login('lpci@uisee.com', 'lpci2021@03')
        print('login')
        smtp.sendmail(sender, receiver, msg.as_string())
        print('send successfully')
    except smtplib.SMTPException:
        print('Error Mail Send ')

def send_mail_personal(sender, receiver, content_html, attachments, subject):

    # read html
    content = content_html
    # with open(content_html, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         content += line

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ','.join(receiver)
    msg['subject'] = subject
    msg.attach(MIMEText(content, 'html', 'utf-8'))

    for attachment in attachments:
        if attachment is None:
            continue
        with open(attachment, 'rb') as fp:
            ctype, encoding = mimetypes.guess_type(attachment)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            sfile = MIMEBase(maintype, subtype)
            sfile.set_payload(fp.read())
        encoders.encode_base64(sfile)
        #sfile['Content-Type'] = 'application/octet-stream'
        sfile['Content-Disposition'] = 'attachment;filename = {}'.format(os.path.basename(attachment))
        msg.attach(sfile)

    try:
        # login
        smtp = smtplib.SMTP_SSL('smtp.feishu.cn', 465)
        # need information
        smtp.login('lpci@uisee.com', 'IFX6O7PVPTqYoB4e')
        print('login')
        smtp.sendmail(sender, receiver, msg.as_string())
        print('send successfully')
        return True
    except smtplib.SMTPException:
        print('Error Mail Send ')
    return False


'''main'''
if __name__ == '__main__':

    send_mail_personal('lpci@uisee.com', ["xiaoqiang.cheng@uisee.com"], "task_info/cvt_console_log.out",
                ["model_deploy_save/test.onnx.1080ti.trt4.bin",
                    "model_deploy_save/test.onnx.1080ti.trt8.bin"],
                "test")


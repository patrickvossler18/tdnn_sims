import sys, json, base64, datetime
import boto3

ec2_client = boto3.client("ec2")
ENCODING = "utf-8"

for setting_num in ["2"]:
    for p in [3, 5, 10, 15, 20]:
        # read in the launch template as a dictionary
        with open(f"setting_{setting_num}_launch_template.json", "r") as fp:
            setting_template = json.load(fp)

        # read in the bash script
        with open(f"setting_{setting_num}_startup.sh", "r") as fp:
            setting_startup = fp.readlines()
        # change DIM=
        setting_startup[1] = f"DIM={p}\n"
        setting_startup_bytes = bytes("".join(setting_startup), ENCODING)
        setting_startup_b64 = base64.b64encode(setting_startup_bytes).decode(ENCODING)
        setting_template["UserData"] = setting_startup_b64
        setting_template["TagSpecifications"][0]["Tags"][0][
            "Value"
        ] = f"tdnn setting {setting_num} p = {p}"
        # launch_info_fp = f"setting_{setting_num}_{p}_launch_info.json"
        # with open(launch_info_fp, "w") as fp:
        #     json.dump(setting_template, fp)
        lt = ec2_client.create_launch_template(
            LaunchTemplateName=f"setting_{setting_num}_{p}_template_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}",
            LaunchTemplateData=setting_template,
        )

        lt_specifics = {"LaunchTemplateId": lt["LaunchTemplate"]["LaunchTemplateId"]}

        launchedInstances = ec2_client.run_instances(
            MaxCount=1, MinCount=1, LaunchTemplate=lt_specifics
        )
        print(launchedInstances)

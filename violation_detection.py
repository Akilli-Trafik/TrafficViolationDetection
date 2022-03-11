import json
import sys
import socket
import numpy as np
from confluent_kafka import DeserializingConsumer, SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import StringDeserializer, StringSerializer
from violation1 import start_violation_detection_for_video
from aws_s3 import upload_to_s3_bucket

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


class Video:
    def __init__(self, video_data, video_name, bboxes_data):
        self.video_data = video_data
        self.video_name = video_name
        self.bboxes_data = bboxes_data

    def export_here(self, video_id):
        output_video_path = "./received_output/{}.mp4".format(video_id)
        output_video_bbox_path = "./received_output/{}.txt".format(video_id)
        bbox_json = json.dumps(self.bboxes_data)
        if not(self.video_data is None):
            new_video_file = open(output_video_path, 'xb')
            new_video_bbox_file = open(output_video_bbox_path, 'x')
            new_video_file.write(self.video_data)
            new_video_bbox_file.write(bbox_json)
            new_video_file.close()
            new_video_bbox_file.close()


def dict_to_video(dict_obj, ctx):
    if dict_obj is None:
        return None
    return Video(dict_obj['video_data'], dict_obj['video_name'], dict_obj['bboxes_data'])


class Violation:
    def __init__(self, violation_video_url):
        self.violation_video_url = violation_video_url


def violation_to_dict(violation_obj, ctx):
    if violation_obj is None:
        return None
    return {"violation_video_url": violation_obj.violation_video_url}


def delivery_report(err, msg):
    if err is not None:
        print("Delivery failed for User record {}: {}".format(msg.key(), err))
        return
    print('User record {} successfully produced to {} [{}] at offset {}'.format(
        msg.key(), msg.topic(), msg.partition(), msg.offset()))


deserializer_schema_str = """
    {
        "namespace": "confluent.io.examples.serialization.avro",
        "name": "Video",
        "type": "record",
        "fields": [
            {"name": "video_data", "type": "bytes"},
            {"name": "video_name", "type": "string"},
            {"name": "bboxes_data", "type": ["null", {"type":"array", "items":{"type":"array", "items":{"type":"array","items":"int"}}, "default": []}]}
        ]
    }
    """

serializer_schema_str = """
{
    "namespace": "confluent.io.examples.serialization.mongodb",
    "name": "Violation",
    "type": "record",
    "fields": [
        {"name": "violation_video_url", "type":"string"}
    ]
}
"""

schema_registry_conf = {'url': 'http://localhost:8081'}
schema_registry_client = SchemaRegistryClient(schema_registry_conf)
avro_deserializer = AvroDeserializer(schema_str=deserializer_schema_str,
                                     schema_registry_client=schema_registry_client,
                                     from_dict=dict_to_video)
avro_serializer = AvroSerializer(schema_str=serializer_schema_str,
                                 schema_registry_client=schema_registry_client,
                                 to_dict=violation_to_dict)

string_deserializer = StringDeserializer('utf_8')
string_serializer = StringSerializer('utf_8')


def main():
    args = sys.argv[1:]

    producer_conf = {'bootstrap.servers': 'localhost:9092,localhost:9092',
                     'client.id': socket.gethostname(),
                     'key.serializer': string_serializer,
                     'value.serializer': avro_serializer}
    consumer_conf = {'bootstrap.servers': 'localhost:9092,localhost:9092',
                     'key.deserializer': string_deserializer,
                     'value.deserializer': avro_deserializer,
                     'group.id': args[0],
                     'auto.offset.reset': "earliest"}
    producer = SerializingProducer(producer_conf)
    consumer = DeserializingConsumer(consumer_conf)
    consumer.subscribe([args[1]])

    while True:
        producer.poll(0.0)
        try:
            # SIGINT can't be handled when polling, limit timeout to 1 second.
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            video_obj = msg.value()
            video_id = msg.key()
            if video_obj is not None:
                print("video record {}: videoDataType: {}\n"
                      "videoName: {} boxes: {}\n"
                      .format(video_id, type(video_obj.video_data),
                              video_obj.video_name, video_obj.bboxes_data))
                video_obj.export_here(video_id)
                violation_name = start_violation_detection_for_video(
                    video_id)  # Burası violation yapılan yer
                if violation_name is not None:
                    video_url = upload_to_s3_bucket(video_id, violation_name)
                    violation_obj = Violation(video_url)
                    producer.produce(
                        "d_topic", key=video_id, value=violation_obj, on_delivery=delivery_report)
        except KeyboardInterrupt:
            break
        except ValueError:
            print("Invalid input, discarding record...")
            continue

    consumer.close()
    producer.flush()


main()

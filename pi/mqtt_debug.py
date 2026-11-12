"""MQTT publisher for the existing sensor topics and the debug topics.

Payloads on ``sensors/imu`` and ``sensors/tof`` match ``publish_sensors.py``
so the PC visualizer does not care which process is running. Debug topics
are extra and only sent when the matching flag is on.
"""

import json
import logging

import paho.mqtt.client as mqtt

from publish_sensors import IMU_TOPIC, TOF_TOPIC

log = logging.getLogger("mqtt_debug")

DEBUG_TOPIC = "debug/pipeline"
AUDIO_TOPIC = "debug/audio"


class MqttDebug:
    def __init__(self, host, port, client_id):
        self.host = host
        self.port = int(port)
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.reconnect_delay_set(min_delay=1, max_delay=10)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self.connected = False

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self.connected = reason_code == 0
        if self.connected:
            log.info("MQTT connected to %s:%s", self.host, self.port)
        else:
            log.error("MQTT refused: %s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connected = False
        if reason_code != 0:
            log.warning("MQTT disconnected (%s)", reason_code)

    def start(self):
        try:
            self._client.connect(self.host, self.port, keepalive=30)
        except Exception:
            log.exception("MQTT connect to %s:%s failed", self.host, self.port)
            raise
        self._client.loop_start()

    def close(self):
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            log.exception("MQTT close")

    def publish(self, topic, payload):
        body = json.dumps(payload, separators=(",", ":"))
        info = self._client.publish(topic, body, qos=0)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            log.warning("publish %s failed rc=%s", topic, info.rc)

    def publish_imu(self, mono_ts_xyz_fake):
        _mono, ts, x, y, z, fake = mono_ts_xyz_fake
        self.publish(IMU_TOPIC, {
            "ts": ts,
            "x": round(x, 4),
            "y": round(y, 4),
            "z": round(z, 4),
            "fake": bool(fake),
        })

    def publish_tof(self, frame, resolution):
        _mono, ts, distances, status, temp, fake, _read_ms = frame
        self.publish(TOF_TOPIC, {
            "ts": ts,
            "resolution": resolution,
            "distances": distances,
            "status": status,
            "silicon_temp_c": temp,
            "fake": bool(fake),
        })

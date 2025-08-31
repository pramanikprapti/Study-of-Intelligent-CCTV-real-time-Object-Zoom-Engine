from onvif import ONVIFCamera
import time

mycam = ONVIFCamera('')

# Create PTZ & Media services
media = mycam.create_media_service()
ptz = mycam.create_ptz_service()

# Get profiles
profiles = media.GetProfiles()
profile=profiles[0]
ptz_status=ptz.GetStatus({'ProfileToken':profile.token})

print('Allis okk')
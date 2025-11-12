# credit to Yu Qiao for specifying this mapping!

import logging.config


logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")

def get_combined_anatomy_site_category(anatomy_sites: list[str]) -> str:
   """
   This is a helper function to map the anatomy sites from the INTERNAL and the BTXRD dataset to a common categorization.
   It takes anatomy sites from either dataset and returns a common category.
   It accepts a list, since in the BTXRD dataset sometimes multiple anatomy sites are given.
   """

   if len(anatomy_sites) == 0:
      logger.critical("Anatomy sites list is empty. Cannot determine combined anatomy site category.")
      raise ValueError("Anatomy sites list cannot be empty.")

   mapping = {
    "Clavicula": "shoulder",
    "Scapula": "shoulder",
    "shoulder-joint": "shoulder",

    "Humerus": "upper arm",
    "humerus": "upper arm",
    "humerus, shoulder-joint": "upper arm",

    "elbow-joint": "elbow",
    "Ulna": "lower arm",
    "ulna": "lower arm",
    "Radius": "lower arm",
    "radius": "lower arm",
    "ulna, radius": "lower arm",
    "hand, radius": "lower arm",
    "hand, ulna, radius": "lower arm",

    "hand": "hand",
    "wrist-joint": "hand",
    "Manus": "hand",

    "Columna vertebralis": "spine",

    "Os pubis": "hip",
    "Os ischii": "hip",
    "Os sacrum": "hip",
    "Os ilium": "hip",
    "hip-joint": "hip",
    "hip bone": "hip",
    "hip bone, hip-joint": "hip",

    "Femur": "upper leg",
    "femur": "upper leg",
    "femur, hip bone": "upper leg",

    "Patella": "knee",
    "knee-joint": "knee",

    "Tibia": "lower leg",
    "Fibula": "lower leg",
    "tibia": "lower leg",
    "fibula": "lower leg",
    "ankle-joint": "lower leg",
    "tibia, fibula": "lower leg",
    "foot, tibia, fibula": "lower leg",

    "Pes": "foot",
    "foot": "foot",
    "foot, ankle-joint": "foot",

    "tibia, fibula, femur": "leg",
    "tibia, femur": "leg",
    "fibula, femur": "leg",
    "tibia, fibula, femur, hip bone": "leg",
    "tibia, fibula, hip bone": "leg",

    "ulna, radius, humerus": "arm",
    "ulna, humerus": "arm",
    "radius, humerus": "arm",
   }

   try:
      if len(anatomy_sites) == 0:
         return mapping[anatomy_sites[0]]
      else:
         anatomy_site = ", ".join(anatomy_sites)
         return mapping[anatomy_site]
   except KeyError:
      logger.critical(f"Anatomy site '{anatomy_site}' not found in mapping.")
      raise
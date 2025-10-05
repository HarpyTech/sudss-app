import mongoengine as me
from datetime import datetime

class Report(me.Document):
    """
    MongoEngine Document model for storing diagnosis reports.
    """
    generated_date = me.DateTimeField(default=datetime.utcnow)
    category = me.StringField(required=True, choices=['X-Ray', 'MRI', 'General', 'Lab'])
    input_type = me.StringField(required=True, choices=['Image', 'Text'])
    trust_score = me.FloatField(min_value=0, max_value=100, default=0.0)
    summary = me.StringField(required=True)

    def to_dict(self):
        """
        Converts the MongoEngine document to a dictionary for API responses.
        """
        return {
            "id": str(self.id),
            "date": self.generated_date.strftime("%Y-%m-%d, %I:%M %p"),
            "category": self.category,
            "inputType": self.input_type,
            "trustScore": self.trust_score,
            "summary": self.summary
        }

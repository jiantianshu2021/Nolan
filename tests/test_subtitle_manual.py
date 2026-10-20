from copy import deepcopy
import unittest
from norlan.subtitle_sync import make_manual_report, manual_report_matches, report_matches

class ManualAdoptionTests(unittest.TestCase):
    def test_manual_adoption_is_bound_and_never_claims_machine_pass(self):
        caps=[{'start':1.,'end':2.,'text':'手动字幕'}]
        report=make_manual_report(caps,audio_sha256='a'*64,source_key='s'*64,revision=2,manual_record_hash='b'*64)
        self.assertTrue(manual_report_matches(report,caps,audio_sha256='a'*64,source_key='s'*64))
        self.assertFalse(report_matches(report,caps))
        self.assertFalse(manual_report_matches(report,[{**caps[0],'start':.9}]))
        self.assertFalse(manual_report_matches(report,caps,audio_sha256='c'*64))
        self.assertFalse(manual_report_matches(report,caps,source_key='changed'))
        for key,value in [('status','passed'),('independent_evidence',True),('revision',True),('manual_record_hash','')]:
            changed=deepcopy(report);changed[key]=value
            self.assertFalse(manual_report_matches(changed,caps))

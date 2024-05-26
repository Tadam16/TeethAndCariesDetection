import 'package:detection_app/services/dio/dicom_api_service.dart';
import 'package:dio/dio.dart';

import 'package:retrofit/dio.dart';

class Interactor {

  final  DicomApiService _service;
  Interactor(this._service);


  Future<dynamic> postPhoto(MultipartFile image) async {
    HttpResponse<dynamic> response = await _service.upload({image.filename!:image});
    return response.data;
  }


}

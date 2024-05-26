import 'dart:async';
import 'package:dio/dio.dart';
import 'package:retrofit/retrofit.dart';

/// Abstract class for an Api calls collection

abstract class DicomApi {

  Future<HttpResponse<dynamic>> upload(Map<String, MultipartFile> photos);

}
